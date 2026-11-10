from __future__ import annotations

import math
import os
import pickle
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import tyro
import viser
from scipy.spatial.transform import Rotation as R

from rtt_utils.constants import TABLE_HEIGHT
from rtt_utils.humans import thin_frames
from rtt_utils.paths import get_home_path
from rtt_utils.skeleton import bones_from_joints_21
from rtt_utils.video import get_frames_from_video
from ttg.types.tt4d import TT4D
from visualization.viser_utils import (
    BLUE,
    add_foot_contact,
    add_foot_vertex_spheres,
    add_handedness_marker,
    add_smpl_animation_ghosts,
    add_video_and_camera_to_viser,
    make_ball_and_trail_pool,
    render_grid_floor,
    render_human_skeleton,
    render_table,
    solid_color,
    update_ball_and_trail,
)


@dataclass
class Config:
    """Viser visualization for table tennis reconstruction."""

    gameplay_dir: str = (
        f"{get_home_path()}/pipeline_outputs/shared/batch_clipped_long_1/7qJgWxBCN48_131950_132665_0_4_1_9"
    )
    """Path to the gameplay directory containing reconstruction data."""
    tt4d_file: str = ""
    """Path to a pickled TT4D (e.g. one under pipeline_outputs/tt4d_pickles/). Overrides gameplay_dir;
    the video is read from the pickle's own gameplay_dir when that folder still exists."""
    lite: bool = False
    """Skip loading the video (and the camera image plane)."""
    pose_source: str = "4dhumans"
    """Which human poses to load: "genmo" (singles only) or "4dhumans"."""
    port: int = 8080
    """Port the viewer is served on."""


def load_tt4d(gameplay_dir: str, tt4d_file: str = "", lite: bool = False, pose_source: str = "4dhumans"):
    """Load a clip's TT4D and unpack what the scene needs: ball track, players, camera, video frames, fps."""
    if tt4d_file:
        with open(tt4d_file, "rb") as f:
            tt4d = pickle.load(f)
        gameplay_dir = getattr(tt4d, "gameplay_dir", None)
    else:
        tt4d = TT4D.from_gameplay_dir(gameplay_dir, None, pose_source=pose_source)
        if hasattr(tt4d, "handedness") and tt4d.handedness is not None:
            tt4d.translation_fix_wrist_ball()

        print(tt4d.failure_reason)

    video_file = None
    if not lite and gameplay_dir:
        clip_name = gameplay_dir.rstrip("/").split("/")[-1]
        candidate = os.path.join(gameplay_dir, "video_second_clipped_only_tracknet", f"{clip_name}.mp4")
        if os.path.exists(candidate):
            video_file = candidate
        else:
            print(f"Video not found at {candidate}; showing the scene without video.")
    vis = tt4d.ball_2d[:, 1].astype(np.float32).reshape(-1, 1)  # 1=visible, 0=invisible
    vis[:] = 1

    ball_positions = np.hstack((vis, tt4d.ball_3d))

    if getattr(tt4d, "label", None) == "doubles":
        # tt4d.humans / tt4d.skeletons only expose the first two players; show all four
        player_smpls = [np.array(h) for h in tt4d._humans]
        skeletons = tt4d._skeletons
    else:
        player_smpls = [np.array(tt4d.humans[0]), np.array(tt4d.humans[1])]
        skeletons = tt4d.skeletons  # list of 2 arrays, each (T, 21, 3)

    # Camera pose → Viser
    rvec = np.asarray(tt4d.rvecs[0]).reshape(3)
    tvec = np.asarray(tt4d.tvecs[0]).reshape(3, 1)
    R_cw, _ = cv2.Rodrigues(rvec)
    R_wc = R_cw.T
    cam_pos = (-R_cw.T @ tvec).ravel()
    q_xyzw = R.from_matrix(R_wc).as_quat()
    cam_wxyz = np.concatenate([[q_xyzw[3]], q_xyzw[:3]])

    cam_pos = cam_pos / 1.5

    if video_file is None:
        frames, num_frames, fps = None, len(ball_positions), getattr(tt4d, "fps", 30)
    else:
        frames = get_frames_from_video(video_file)
        frames = thin_frames(frames, getattr(tt4d, "dropped_frames", None))
        if tt4d.indices_relative_to_video_second_clipped_only_tracknet is not None:
            frames = frames[slice(*tt4d.indices_relative_to_video_second_clipped_only_tracknet)]
        num_frames = min(len(frames), len(ball_positions))
        fps = tt4d.fps  # thinned rate, not the video container's

    handedness = getattr(tt4d, "handedness", None)
    foot_contact = getattr(tt4d, "foot_contact", None)
    if foot_contact is None:
        # the 4DHumans singles loader does not set foot contact
        foot_contact = np.zeros((num_frames, 2), dtype=np.float32)

    return (
        ball_positions,
        player_smpls,
        foot_contact,
        cam_pos,
        cam_wxyz,
        frames,
        fps,
        num_frames,
        skeletons,
        handedness,
    )


def main(cfg: Config):
    """Serve the viser scene for one clip and animate it until interrupted."""
    (
        ball_positions,
        player_smpls,
        foot_contact,
        cam_pos,
        cam_wxyz,
        video,
        fps,
        num_frames,
        skeletons,
        handedness,
    ) = load_tt4d(cfg.gameplay_dir, cfg.tt4d_file, cfg.lite, cfg.pose_source)

    server = viser.ViserServer(port=cfg.port)
    server.scene.set_up_direction("+z")
    server.scene.add_frame(
        "/world_frame",
        position=(0, 0, TABLE_HEIGHT),
        axes_length=0.15,
        axes_radius=0.005,
    )
    render_grid_floor(server)
    render_table(server)
    ball_h, pool = make_ball_and_trail_pool(server, "ball")

    gui_timestep = server.gui.add_slider("Frame", min=0, max=num_frames - 1, step=1, initial_value=0)
    gui_next = server.gui.add_button("Next Frame")
    gui_prev = server.gui.add_button("Prev Frame")
    gui_play = server.gui.add_checkbox("Play", True)
    gui_framerate = server.gui.add_slider("FPS", min=1.0, max=60.0, step=0.1, initial_value=fps)

    @gui_next.on_click
    def _(_):
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev.on_click
    def _(_):
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    @gui_play.on_update
    def _(_):
        gui_timestep.disabled = gui_play.value

    if video is not None:
        add_video_and_camera_to_viser(
            server,
            gui_timestep,
            video,
            cam_wxyz,
            cam_pos,
            max_frames=1500,
            subsample=1,
            img_maxsize=1280,
        )

    if len(player_smpls) != 0:
        add_smpl_animation_ghosts(
            server,
            player_smpls=player_smpls,
            gui_timestep=gui_timestep,
            subsample=1,
            max_keep=1,
            poses_are_rot_mats=True,
        )

        for pid in range(len(player_smpls)):
            fc = foot_contact[pid] if isinstance(foot_contact, list) else foot_contact
            add_foot_contact(server, fc, gui_timestep, player_smpls[pid], prefix=f"p{pid}")
            add_foot_vertex_spheres(server, gui_timestep, player_smpls[pid], prefix=f"p{pid}")

        if handedness is not None:
            for pid in range(len(player_smpls)):
                add_handedness_marker(
                    server,
                    handedness[pid],
                    gui_timestep,
                    player_smpls[pid],
                    prefix=f"p{pid}/handedness",
                )

    # Skeleton visualization (21-joint BVH / cory337 skeletons)
    if skeletons is not None and len(skeletons) >= 2:
        skel_colors = [
            np.array((0, 120, 255), np.uint8),
            np.array((0, 255, 80), np.uint8),
            np.array((255, 160, 0), np.uint8),
            np.array((255, 60, 200), np.uint8),
        ]
        skeleton_handles = []
        for pid in range(len(skeletons)):
            color = skel_colors[pid % len(skel_colors)]
            joint_handle, bone_handle = render_human_skeleton(server, player_name=f"skeleton/player_{pid}", color=color)
            skeleton_handles.append((skeletons[pid], joint_handle, bone_handle))

        def _update_skeletons(t: int):
            for pos_seq, joint_h, bone_h in skeleton_handles:
                frame_idx = min(t, len(pos_seq) - 1)
                sk = pos_seq[frame_idx]
                if np.isfinite(sk).all():
                    joint_h.points = sk
                    joint_h.colors = solid_color(BLUE, 21)[:, 0, :]
                    bone_h.points = bones_from_joints_21(sk)

        _update_skeletons(0)
        gui_timestep.on_update(lambda _: _update_skeletons(int(gui_timestep.value)))

    @server.on_client_connect
    def _(c: viser.ClientHandle):
        global active_client
        active_client = c

        if cam_pos is not None:
            c.camera.position = cam_pos
            c.camera.wxyz = cam_wxyz

        c.camera.look_at = (0.0, 0.0, TABLE_HEIGHT)
        c.camera.fov = math.radians(60)
        c.camera.near, c.camera.far = 1e-3, 2e3

    print("Setup complete. Animating.")

    trail = deque(maxlen=10)
    last_idx = None
    while True:
        idx = int(gui_timestep.value)
        if idx == 0:
            trail.clear()
            update_ball_and_trail(ball_h, pool, [1000, 1000, 1000], trail)  # don't want the ball to be visible

        # Ball update
        if ball_positions[gui_timestep.value, 0] and idx != last_idx:
            p = ball_positions[gui_timestep.value, 1:4]
            if np.abs(p).sum() != 0:
                update_ball_and_trail(ball_h, pool, p, trail)
                last_idx = idx

        # advance only if playing
        if gui_play.value:
            gui_timestep.value = (idx + 1) % num_frames

        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
