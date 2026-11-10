"""Viser scene helpers shared by the reconstruction visualizers: table, ball trail, skeletons, SMPL meshes."""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import viser
from scipy.spatial.transform import Rotation as R

from rtt_utils.constants import NET_HALF_LENGTH, NET_HEIGHT, TABLE_HALF_LENGTH, TABLE_HALF_WIDTH, TABLE_HEIGHT
from rtt_utils.skeleton import N_BONES_21
from rtt_utils.smpl import get_smpl_sole_indices, load_smpl_model, smpl_forward_chunked

# Default player colors for SMPL mesh visualization
PLAYER_COLORS = [
    (0.30, 0.85, 1.00),  # left player - cyan
    (0.85, 0.55, 0.50),  # right player - salmon
    (0.55, 0.85, 0.60),  # player 3 - green
    (0.60, 0.60, 0.95),  # player 4 - purple
]

BLUE = np.array((0, 120, 255), np.uint8)  # left player
GREEN = np.array((0, 255, 80), np.uint8)  # right player


def render_grid_floor(server):
    """Add an 8 m x 8 m grid of 10 cm cells on the floor (z = 0)."""
    grid_lines = [[[-40, n, 0], [40, n, 0]] for n in range(-40, 40, 1)] + [
        [[n, -40, 0], [n, 40, 0]] for n in range(-40, 40, 1)
    ]
    grid_lines = np.asarray(grid_lines, dtype=float) / 10

    server.scene.add_line_segments("/ground_grid", points=grid_lines, colors=(200, 200, 200), line_width=1)


def render_ball(server, name):
    """Add a red ball sphere and an empty point-cloud trail; returns (ball_handle, trail_handle)."""
    BASE_GRAY = 240

    ball_handle = server.scene.add_icosphere(f"/ball/{name}", radius=20 * 1e-3, color=(255, 0, 0))
    trail_handle = server.scene.add_point_cloud(
        "/ball/trail",  # fading streak
        points=np.empty((0, 3)),  # start empty
        colors=np.array((BASE_GRAY, BASE_GRAY, BASE_GRAY), np.uint8),
        point_size=32.0 * 1e-3,  # 8 mm screen size
    )

    return ball_handle, trail_handle


def make_ball_and_trail_pool(server, name="ball"):
    """Add the ball sphere plus a pool of hidden spheres reused for its fading trail; returns (ball, pool)."""
    trail_max = 48  # how many trail elements to keep visible
    BASE_GRAY = 240  # white-ish
    TAIL_R = 36.0 * 1e-3  # tail

    # main ball (single sphere)
    ball_h = server.scene.add_icosphere(
        f"/ball/{name}", radius=20 * 1e-3, color=(255, 255, 255), material="toon3", flat_shading=True
    )
    # pre-allocate N tiny spheres for the streak
    pool = []
    for k in range(trail_max):
        h = server.scene.add_icosphere(
            f"/ball/trail/{k}",
            radius=TAIL_R,  # will be updated each frame
            color=(BASE_GRAY, BASE_GRAY, BASE_GRAY),
            opacity=1.0,
            material="toon3",
            flat_shading=True,
            visible=False,  # hidden until used
        )
        pool.append(h)
    return ball_h, pool


def update_ball_and_trail(ball_h, pool, p, trail):
    """Move the ball to `p`, push `p` onto the trail deque and redraw the pool spheres along it."""
    BASE_GRAY = 240  # white-ish
    HEAD_R = 36.0 * 1e-3  # near the ball
    TAIL_R = 36.0 * 1e-3  # tail

    # head (big ball)
    ball_h.position = np.asarray(p, np.float32)

    # update trail buffer (newest at index 0)
    trail.appendleft(np.asarray(p, np.float32))
    pts = list(trail)
    n = len(pts)

    # precompute fade head→tail for radius + brightness
    if n:
        w = np.linspace(1.0, 0.3, n, dtype=np.float32)  # 1.0 head … 0.3 tail
        radii = (TAIL_R + (HEAD_R - TAIL_R) * w).astype(np.float32)
        cols = (BASE_GRAY * w).astype(np.uint8)

    # place spheres
    for k, h in enumerate(pool):
        if k < n:
            h.visible = True
            h.position = pts[k]
            h.radius = float(radii[k])
            g = int(cols[k])
            h.color = (g, g, g)  # RGB
        else:
            h.visible = False


def render_table(server):
    """Draw the table top outline, its four legs and the net as line segments; returns TABLE_HEIGHT."""
    # ---- table top outline ----
    table_xy = np.array(
        [
            [-TABLE_HALF_LENGTH, -TABLE_HALF_WIDTH],
            [-TABLE_HALF_LENGTH, TABLE_HALF_WIDTH],
            [TABLE_HALF_LENGTH, TABLE_HALF_WIDTH],
            [TABLE_HALF_LENGTH, -TABLE_HALF_WIDTH],
        ]
    )
    table_segs = np.stack(
        [
            np.column_stack([table_xy, np.full(4, TABLE_HEIGHT)]),
            np.column_stack([np.roll(table_xy, -1, 0), np.full(4, TABLE_HEIGHT)]),
        ],
        axis=1,
    )

    # ---- net: two posts + top & bottom tape ----
    H = NET_HEIGHT
    net_segs = np.array(
        [
            [[0, NET_HALF_LENGTH, TABLE_HEIGHT], [0, NET_HALF_LENGTH, TABLE_HEIGHT + H]],  # left post
            [[0, -NET_HALF_LENGTH, TABLE_HEIGHT], [0, -NET_HALF_LENGTH, TABLE_HEIGHT + H]],  # right post
            [[0, NET_HALF_LENGTH, TABLE_HEIGHT + H], [0, -NET_HALF_LENGTH, TABLE_HEIGHT + H]],  # top tape
            [[0, NET_HALF_LENGTH, TABLE_HEIGHT], [0, -NET_HALF_LENGTH, TABLE_HEIGHT]],  # bottom edge
        ]
    )

    # ---- four table legs ----
    table_legs = np.array(
        [
            [[-TABLE_HALF_LENGTH, -TABLE_HALF_WIDTH, TABLE_HEIGHT], [-TABLE_HALF_LENGTH, -TABLE_HALF_WIDTH, 0]],
            [[-TABLE_HALF_LENGTH, TABLE_HALF_WIDTH, TABLE_HEIGHT], [-TABLE_HALF_LENGTH, TABLE_HALF_WIDTH, 0]],
            [[TABLE_HALF_LENGTH, TABLE_HALF_WIDTH, TABLE_HEIGHT], [TABLE_HALF_LENGTH, TABLE_HALF_WIDTH, 0]],
            [[TABLE_HALF_LENGTH, -TABLE_HALF_WIDTH, TABLE_HEIGHT], [TABLE_HALF_LENGTH, -TABLE_HALF_WIDTH, 0]],
        ]
    )

    server.scene.add_line_segments("/table/top", points=table_segs, colors=(20, 120, 20), line_width=3)
    server.scene.add_line_segments("/table/legs", points=table_legs, colors=(100, 60, 20), line_width=4)
    server.scene.add_line_segments("/table/net_posts", points=net_segs[:2], colors=(20, 20, 160), line_width=3)
    server.scene.add_line_segments("/table/net_tape", points=net_segs[2:], colors=(240, 240, 240), line_width=2)

    return TABLE_HEIGHT


def solid_color(rgb, n):
    """Repeat one RGB color into the (n, 2, 3) color array viser expects for n line segments."""
    return np.broadcast_to(rgb, (n, 2, 3)).astype(np.uint8)


def render_human_skeleton(server, player_name: str, color: np.ndarray):
    """Add empty joint and bone handles for a 21-joint skeleton; fill them per frame with bones_from_joints_21."""
    joint_handle = server.scene.add_point_cloud(
        f"/{player_name}/joints",
        points=np.empty((0, 3)),
        colors=np.empty((0, 3), np.uint8),
        point_size=0.02,
    )
    bone_handle = server.scene.add_line_segments(
        f"/{player_name}/bones",
        points=np.empty((N_BONES_21, 2, 3)),
        colors=solid_color(color, N_BONES_21),
        line_width=2.0,
    )
    return joint_handle, bone_handle


def orbit_about_z(active_client, angle_rad, cam_pos_m, cam_wxyz, center=(0.0, 0.0, TABLE_HEIGHT)):
    """Rotate the client's camera by angle_rad about the vertical axis through `center`."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    if active_client is None:
        return

    # ---- translate–rotate–translate back (position) ----
    active_client.camera.position = Rz @ (cam_pos_m - center) + center

    # ---- orientation (extrinsic world-Z rotation) ----
    R_wc = viser.transforms.SO3(cam_wxyz).as_matrix()
    R_wc = Rz @ R_wc  # change the orientation of the camera relative to the world frame
    q_xyzw = R.from_matrix(R_wc).as_quat()  # SciPy → x,y,z,w. Viser expects R_wc
    active_client.camera.wxyz = [q_xyzw[3], *q_xyzw[:3]]  # w,x,y,z order


def add_video_and_camera_to_viser(
    server: viser.ViserServer,
    gui_timestep,
    video,
    cam_wxyz: np.ndarray,
    cam_pos: np.ndarray,
    max_frames: int = 1500,
    subsample: int = 1,
    img_maxsize: int = 1280,
):
    """
    Reads up to max_frames from video_file (downsampled by `subsample`), then
    adds each frame as a textured frustum plus a coordinate axes at cam_pos_mm/cam_wxyz.
    """
    assert video is not None

    if isinstance(video, str):
        # 1) load frames
        cap = cv2.VideoCapture(video)
        images = []
        idx = 0
        while len(images) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % subsample == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(rgb)
            idx += 1
        cap.release()
    elif isinstance(video, list):
        images = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in video]
    else:
        raise Exception(f"Invalid format for video information: {type(video)}.")

    num_frames = len(images)
    if num_frames == 0:
        raise RuntimeError("No frames found in video.")

    # 2) create per-frame nodes
    frame_nodes = []
    for i, img in enumerate(images):
        # optional downscale
        h, w = img.shape[:2]
        if max(h, w) > img_maxsize:
            scale = img_maxsize / max(h, w)
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # camera frustum textured with this frame
        fr = server.scene.add_camera_frustum(
            f"/video/frustum{i}",
            fov=math.radians(60),
            aspect=img.shape[1] / img.shape[0],
            scale=600 * 1e-3,
            line_width=1.5,
            wxyz=cam_wxyz,
            position=cam_pos,
            image=img,
        )

        # little coordinate axes at the same pose
        ax = server.scene.add_frame(
            f"/video/axes{i}",
            wxyz=cam_wxyz,
            position=cam_pos,
            axes_length=150 * 1e-3,
            axes_radius=5 * 1e-3,
        )

        frame_nodes.append((fr, ax))

    # 3) helper to show exactly one frame
    def show_frame(n):
        for idx, (fr, ax) in enumerate(frame_nodes):
            vis = idx == n
            fr.visible = vis
            ax.visible = vis

    show_frame(0)

    @gui_timestep.on_update
    def _(_):
        show_frame(gui_timestep.value)


def add_smpl_animation_ghosts(
    server: viser.ViserServer,
    player_smpls: list[np.ndarray],
    gui_timestep,
    *,
    smpl_model=None,
    model_path: str = None,
    gender: str = "male",
    subsample: int = 1,
    max_keep: int = 9,
    poses_are_rot_mats: bool = True,
    scale_to_mm: bool = False,
    player_colors: list[tuple] = None,
):
    """
    Animate SMPL meshes with fading ghost trail effect.

    Uses add_mesh_simple() with opacity for clean fading ghosts.

    Args:
        server: Viser server instance
        player_smpls: List of SMPL parameter arrays, one per player.
                     Each array has shape (num_frames,) with dict entries containing
                     'global_orient', 'body_pose', 'betas', 'transl'.
        gui_timestep: Viser GUI slider handle for frame control
        smpl_model: Pre-loaded SMPL model (smplx.SMPL). If None, loads using smplx.create().
        model_path: Path to directory containing SMPL model files.
                   If None and smpl_model is None, uses default assets/smpl path.
        gender: SMPL model gender ('male', 'female', 'neutral'). Only used if loading model.
        subsample: Frame subsampling factor
        max_keep: Number of ghost frames to keep visible
        poses_are_rot_mats: If True, poses are rotation matrices; if False, axis-angle
        scale_to_mm: If True, multiply vertices by 1000*S to convert m -> scene units
        player_colors: Optional list of RGB tuples (0-1 range) for each player
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model if not provided
    if smpl_model is None:
        smpl = load_smpl_model(model_path=model_path, gender=gender, device=str(device))
    else:
        smpl = smpl_model
    faces = np.asarray(smpl.faces, dtype=np.int32, order="C")

    colors = player_colors if player_colors is not None else PLAYER_COLORS

    mesh_cache: dict[int, list[viser.MeshHandle]] = {}

    def to_batch(x_np):
        t = torch.from_numpy(np.asarray(x_np)).float().to(device)
        return t.reshape(1, *t.shape)

    def build_frame_meshes(frame_idx: int) -> list[viser.MeshHandle]:
        if frame_idx in mesh_cache:
            return mesh_cache[frame_idx]

        handles: list[viser.MeshHandle] = []
        for pid, smpl_arr in enumerate(player_smpls):
            if frame_idx >= smpl_arr.shape[0]:
                continue

            entry = smpl_arr[frame_idx]
            go, bp, betas, transl = entry["global_orient"], entry["body_pose"], entry["betas"], entry["transl"]

            betas_t = to_batch(betas)
            transl_t = to_batch(transl)

            go_t = to_batch(go)
            bp_t = to_batch(bp)
            with torch.no_grad():
                out = smpl(
                    betas=betas_t, global_orient=go_t, body_pose=bp_t, transl=transl_t, pose2rot=not poses_are_rot_mats
                )

            verts_m = out.vertices[0].cpu().numpy()
            verts_m = verts_m.astype(np.float32, order="C")

            base_rgb = colors[pid % len(colors)]
            mh = server.scene.add_mesh_simple(
                f"/smpl/p{pid}/frame{frame_idx}",
                vertices=verts_m,
                faces=faces,
                color=base_rgb,
                opacity=1.0,
                flat_shading=False,
                side="double",
                cast_shadow=False,
                receive_shadow=False,
                visible=False,
            )
            handles.append(mh)

        mesh_cache[frame_idx] = handles
        return handles

    build_frame_meshes(0)
    for h in mesh_cache[0]:
        h.visible = True

    @gui_timestep.on_update
    def _(_ev):
        f = gui_timestep.value // subsample
        window = list(range(max(0, f - max_keep + 1), f + 1))
        window_set = set(window)

        for w in window:
            build_frame_meshes(w)

        for fr, handles in mesh_cache.items():
            vis = fr in window_set
            for h in handles:
                h.visible = vis

        if window:
            for i, fr in enumerate(window):
                op = 0.2 + 0.8 * ((i + 1) / len(window))
                for h in mesh_cache[fr]:
                    h.opacity = float(op)

        server.flush()

    return mesh_cache


def add_foot_contact(
    server: viser.ViserServer,
    foot_contact: np.ndarray,
    gui_timestep,
    player_smpl: np.ndarray,
    smpl_model=None,
    model_path: str = None,
    gender: str = "male",
    radius: float = 0.05,
    prefix: str = "foot_contact",
):
    """
    Two icospheres (left / right foot) for player_smpl, colored by foot_contact.

    foot_contact: (T, 2) — col 0 = left foot, col 1 = right foot.
        0 → red (no contact), 1 → green (contact), else → hidden.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl = smpl_model or load_smpl_model(model_path=model_path, gender=gender, device=str(device))

    FOOT_L, FOOT_R = 10, 11
    T = foot_contact.shape[0]
    n = min(T, len(player_smpl))
    foot_pos = np.zeros((T, 2, 3), dtype=np.float32)  # (T, left/right, xyz)
    joints, _ = smpl_forward_chunked(smpl, [player_smpl[t] for t in range(n)], device=device)
    foot_pos[:n, 0] = joints[:, FOOT_L]
    foot_pos[:n, 1] = joints[:, FOOT_R]

    handles = [
        server.scene.add_icosphere(f"/{prefix}/{side}", radius=radius, color=(128, 128, 128), visible=False)
        for side in ("left", "right")
    ]

    def _update(t: int):
        for i, h in enumerate(handles):
            if t >= T:
                h.visible = False
                continue
            val = foot_contact[t, i]
            if val == 1.0:
                h.visible, h.color, h.position = True, (0, 255, 0), foot_pos[t, i]
            elif val == 0.0:
                h.visible, h.color, h.position = True, (255, 0, 0), foot_pos[t, i]
            else:
                h.visible = False

    _update(0)

    @gui_timestep.on_update
    def _(_):
        _update(int(gui_timestep.value))

    return handles


def add_handedness_marker(
    server: viser.ViserServer,
    handedness: str,
    gui_timestep,
    player_smpl: list,
    smpl_model=None,
    model_path: str = None,
    gender: str = "male",
    radius: float = 0.04,
    prefix: str = "handedness",
    color: tuple = (255, 165, 0),
):
    """
    Place an icosphere on the player's dominant hand to indicate handedness.

    Args:
        handedness: "left" or "right".
        Other args mirror add_foot_contact.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl = smpl_model or load_smpl_model(model_path=model_path, gender=gender, device=str(device))

    # SMPL joint indices: left_wrist=20, right_wrist=21
    WRIST_IDX = 20 if handedness == "left" else 21

    T = len(player_smpl)
    joints, _ = smpl_forward_chunked(smpl, [player_smpl[t] for t in range(T)], device=device)
    hand_pos = joints[:, WRIST_IDX]

    handle = server.scene.add_icosphere(
        f"/{prefix}",
        radius=radius,
        color=color,
    )

    def _update(t: int):
        if t < T:
            handle.visible = True
            handle.position = hand_pos[t]
        else:
            handle.visible = False

    _update(0)

    @gui_timestep.on_update
    def _(_):
        _update(int(gui_timestep.value))

    return handle


def add_foot_vertex_spheres(
    server: viser.ViserServer,
    gui_timestep,
    player_smpl: list,
    smpl_model=None,
    radius: float = 0.01,
    prefix: str = "foot_verts",
):
    """Draw the sole vertices of both feet (blue left, yellow right) as small spheres, updated per frame."""
    SOLE_INDICES = get_smpl_sole_indices()
    device = "cuda"
    smpl_model = load_smpl_model(model_path=None, gender="male", device=str(device))

    T = len(player_smpl)
    N = SOLE_INDICES.shape[0] // 2
    _, vertices = smpl_forward_chunked(smpl_model, player_smpl[:T], device=device)
    vertex_positions = vertices[:, SOLE_INDICES, :]

    handles = []
    for i in range(2 * N):
        side = "left" if i < N else "right"
        color = (50, 100, 255) if side == "left" else (255, 200, 0)
        h = server.scene.add_icosphere(f"/{prefix}/{side}_{i}", radius=radius, color=color)
        handles.append(h)

    def _update(t: int):
        if t >= T:
            return
        for i, h in enumerate(handles):
            h.position = vertex_positions[t, i]

    _update(0)

    @gui_timestep.on_update
    def _(_):
        _update(int(gui_timestep.value))

    return handles
