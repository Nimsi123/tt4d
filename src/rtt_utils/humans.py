"""Load a clip's per-frame inputs (ball track, calibration, 4DHumans detections) and pick the players."""

import json
import os
from collections import defaultdict

import cv2
import numpy as np

from rtt_utils.calibration import load_calib, load_H_ground
from rtt_utils.constants import TABLE_LENGTH, TABLE_WIDTH
from rtt_utils.paths import load_file_paths
from rtt_utils.video import get_frames_from_video, get_video_fps


def reindex_4dh_result(start, end, metadata, keypts_2d, keypts_3d, smpl):
    """
    Start and end parameters are relative to a superset video.
    E.g. video_second_clipped and video_second_clipped_only_tracknet.

    Reindexes to _second_clipped_only_tracknet (or whatever is specified by start, end).
    """
    m = []
    k2 = []
    k3 = []
    s = []

    curr_id = 0
    recount = 0
    frame_no = start
    while frame_no <= end:  # inclusive I think?
        while curr_id < len(metadata) and int(metadata[curr_id]['frame']) < frame_no:
            curr_id += 1
        while curr_id < len(metadata) and int(metadata[curr_id]['frame']) == frame_no:
            tmp = metadata[curr_id].copy()
            tmp["frame"] = recount
            m.append(tmp)
            k2.append(keypts_2d[curr_id])
            k3.append(keypts_3d[curr_id])
            s.append(smpl[curr_id])
            curr_id += 1
        frame_no += 1
        recount += 1

    metadata = m
    keypts_2d = np.array(k2)
    keypts_3d = np.array(k3)
    smpl = np.array(s)

    return metadata, keypts_2d, keypts_3d, smpl


def load_4dh(gameplay_dir):
    """
    Tries to load the 4DHumans result for the _only_tracknet second clipped video.

    If this is not available, it returns the 4DHumans result that is present in the folder.
    """
    clip_name = gameplay_dir.split("/")[-1]

    # shortcut (no reindexing needed)
    human_pose_tracker_only_tracknet_dir = os.path.join(gameplay_dir, "human_pose_tracker_only_tracknet", "4DHumans")
    if os.path.exists(os.path.join(human_pose_tracker_only_tracknet_dir, clip_name, f"{clip_name}_keypoints_2d.npy")):
        keypts_2d = np.load(
            os.path.join(human_pose_tracker_only_tracknet_dir, clip_name, f"{clip_name}_keypoints_2d.npy")
        )
        keypts_3d = np.load(
            os.path.join(human_pose_tracker_only_tracknet_dir, clip_name, f"{clip_name}_keypoints_3d.npy")
        )
        smpl = np.load(
            os.path.join(human_pose_tracker_only_tracknet_dir, clip_name, f"{clip_name}_smpl.npy"), allow_pickle=True
        )
        with open(os.path.join(human_pose_tracker_only_tracknet_dir, clip_name, f"{clip_name}_metadata.json")) as f:
            metadata = json.load(f)
        return metadata, keypts_2d, keypts_3d, smpl

    # can compute the _only_tracknet second clipped by reindexing the 4dh result
    human_pose_tracker_dir = os.path.join(gameplay_dir, "human_pose_tracker", "4DHumans")
    clip_indices_path = os.path.join(gameplay_dir, "metadata", "second_clipped_indices.txt")
    clip_indices_only_tracknet_path = os.path.join(gameplay_dir, "metadata_only_tracknet", "second_clipped_indices.txt")
    if os.path.exists(clip_indices_path) and os.path.exists(clip_indices_only_tracknet_path):
        clip_indices = np.loadtxt(clip_indices_path, delimiter=',', max_rows=1)
        clip_indices_only_tracknet = np.loadtxt(clip_indices_only_tracknet_path, delimiter=',', max_rows=1)
        clip_indices[0] <= clip_indices_only_tracknet[0] and clip_indices_only_tracknet[1] <= clip_indices[1]

        keypts_2d = np.load(os.path.join(human_pose_tracker_dir, clip_name, f"{clip_name}_keypoints_2d.npy"))
        keypts_3d = np.load(os.path.join(human_pose_tracker_dir, clip_name, f"{clip_name}_keypoints_3d.npy"))
        smpl = np.load(os.path.join(human_pose_tracker_dir, clip_name, f"{clip_name}_smpl.npy"), allow_pickle=True)
        with open(os.path.join(human_pose_tracker_dir, clip_name, f"{clip_name}_metadata.json")) as f:
            metadata = json.load(f)

        # compute new start and end indices (relative to video_second_clipped)
        start_relative_to_second_clipped = clip_indices_only_tracknet[0] - clip_indices[0]
        end_relative_to_second_clipped = start_relative_to_second_clipped + (
            clip_indices_only_tracknet[1] - clip_indices_only_tracknet[0]
        )
        start_relative_to_second_clipped, end_relative_to_second_clipped = (
            int(start_relative_to_second_clipped),
            int(end_relative_to_second_clipped),
        )

        metadata, keypts_2d, keypts_3d, smpl = reindex_4dh_result(
            start_relative_to_second_clipped, end_relative_to_second_clipped, metadata, keypts_2d, keypts_3d, smpl
        )

        return metadata, keypts_2d, keypts_3d, smpl

    # last resort
    human_pose_tracker_dir = os.path.join(gameplay_dir, "4DHumans")
    if not os.path.exists(human_pose_tracker_dir):
        human_pose_tracker_dir = os.path.join(gameplay_dir, "human_pose_tracker", "4DHumans")
        if not os.path.exists(human_pose_tracker_dir):
            return None, None, None, None

    keypts_2d = np.load(os.path.join(human_pose_tracker_dir, clip_name, f"{clip_name}_keypoints_2d.npy"))
    keypts_3d = np.load(os.path.join(human_pose_tracker_dir, clip_name, f"{clip_name}_keypoints_3d.npy"))
    smpl = np.load(os.path.join(human_pose_tracker_dir, clip_name, f"{clip_name}_smpl.npy"), allow_pickle=True)
    with open(os.path.join(human_pose_tracker_dir, clip_name, f"{clip_name}_metadata.json")) as f:
        metadata = json.load(f)

    return metadata, keypts_2d, keypts_3d, smpl


def thin_frames(x, dropped_frames):
    """Remove the duplicated video frames (indices along axis 0) from an array or list of per-frame data."""
    if dropped_frames is None or len(dropped_frames) == 0:
        return x
    keep = np.ones(len(x), dtype=bool)
    dropped_frames = np.asarray(dropped_frames, dtype=int)
    keep[dropped_frames[dropped_frames < len(x)]] = False
    if isinstance(x, np.ndarray):
        return x[keep]
    return [v for v, k in zip(x, keep) if k]


def thinned_to_video_index(idx, dropped_frames, num_video_frames):
    """Map an index in the thinned sequence back to the index of the same frame in the video."""
    dropped_frames = np.asarray([] if dropped_frames is None else dropped_frames, dtype=int)
    kept = np.setdiff1d(np.arange(num_video_frames), dropped_frames)
    return int(kept[min(idx, len(kept) - 1)])


def thin_4dh_result(dropped_frames, metadata, keypts_2d, keypts_3d, smpl):
    """Drop the 4DHumans detections on duplicated frames and renumber the remaining frames consecutively."""
    num_frames = int(metadata[-1]["frame"]) + 1 if len(metadata) else 0
    keep = np.ones(num_frames, dtype=bool)
    dropped_frames = np.asarray(dropped_frames, dtype=int)
    keep[dropped_frames[dropped_frames < num_frames]] = False
    new_frame = np.cumsum(keep) - 1
    m, k2, k3, s = [], [], [], []
    for i, meta in enumerate(metadata):
        frame = int(meta["frame"])
        if not keep[frame]:
            continue
        tmp = meta.copy()
        tmp["frame"] = int(new_frame[frame])
        m.append(tmp)
        k2.append(keypts_2d[i])
        k3.append(keypts_3d[i])
        s.append(smpl[i])
    return m, np.array(k2), np.array(k3), np.array(s)


def load_task(gameplay_dir, ret_frames=False, thin=True):
    """
    Load everything the reconstruction needs for one clip, aligned to video_second_clipped_only_tracknet.

    Returns the dict from load_file_paths plus fps, calibration (camera_matrix, rvecs, tvecs, H_ground), the
    4DHumans detections moved into the world frame (metadata, keypts_2d, keypts_3d, smpl), the 2D ball track
    and, with thin=True, every per-frame item with the duplicated video frames removed (dropped_frames,
    video_fps, frozen_run).
    """
    paths = load_file_paths(gameplay_dir)
    clip_name = paths["clip_name"]
    video_second_clipped_only_tracknet_file = paths["video_second_clipped_only_tracknet_file"]
    ball_tracker_only_tracknet_file = paths["ball_tracker_only_tracknet_file"]

    fps, _, _ = get_video_fps(video_second_clipped_only_tracknet_file)

    clip_indices_path = paths["metadata_only_tracknet_second_clipped_indices_file"]
    clip_indices = np.loadtxt(clip_indices_path, delimiter=',', max_rows=1)
    only_tracknet_second_clipped_slice = slice(int(clip_indices[0]), int(clip_indices[1]))

    if ret_frames:
        frames = get_frames_from_video(video_second_clipped_only_tracknet_file)
        if len(frames) <= 1:
            print("WHOAH: ", clip_name)
            return ...
    else:
        frames = None

    ball_positions_2d_only_tracknet = np.loadtxt(ball_tracker_only_tracknet_file, delimiter=',', skiprows=1)
    ball_positions_2d_only_tracknet = ball_positions_2d_only_tracknet[
        only_tracknet_second_clipped_slice
    ]  # for x and y, do [:, 2:4]

    rvecs, tvecs, camera_matrix = load_calib(gameplay_dir)
    H_ground = load_H_ground(rvecs, tvecs, camera_matrix)
    metadata, keypts_2d, keypts_3d, smpl = load_4dh(
        gameplay_dir
    )  # attempts to load the only_tracknet_second_clipped slice
    scale_factor = 1.11111111  # 2.22222222222222 # 1.11111111
    keypts_3d, smpl = transform_4dh_to_world_frame(
        metadata, keypts_2d, keypts_3d, smpl, rvecs, H_ground, scale_factor=scale_factor
    )

    # Drop the duplicated video frames found by src/video_thinning (e.g. 25 fps footage stored at 30 fps) from
    # every per-frame feature, so the ball dynamics and the human motion are sampled at the true frame rate.
    video_fps = fps
    frames_to_drop = np.array([], dtype=int)
    frozen_run = 0
    if thin:
        # we assume this applies only for the video_second_clipped_only_tracknet stuff
        with open(paths["video_thinning_frames_to_drop_file"]) as f:
            frames_to_drop = json.load(f)
        frames_to_drop = np.array(frames_to_drop, dtype=int)
        frames_to_drop -= only_tracknet_second_clipped_slice.start
        num_frames = ball_positions_2d_only_tracknet.shape[0]
        frames_to_drop = frames_to_drop[(frames_to_drop >= 0) & (frames_to_drop < num_frames)]  # zero indexed

        if len(frames_to_drop):
            # longest run of consecutive dropped frames: a long run means the video froze, not a frame-rate cadence
            runs = np.split(frames_to_drop, np.where(np.diff(frames_to_drop) != 1)[0] + 1)
            frozen_run = max(len(r) for r in runs)

            ball_positions_2d_only_tracknet = thin_frames(ball_positions_2d_only_tracknet, frames_to_drop)
            ball_positions_2d_only_tracknet[:, 0] = np.arange(len(ball_positions_2d_only_tracknet))  # Frame column
            metadata, keypts_2d, keypts_3d, smpl = thin_4dh_result(frames_to_drop, metadata, keypts_2d, keypts_3d, smpl)
            if frames is not None:
                frames = thin_frames(frames, frames_to_drop)
            fps = fps * (num_frames - len(frames_to_drop)) / num_frames

    ret = {
        **paths,  # all file paths from load_file_paths
        # numeric / data
        "fps": fps,
        "H_ground": H_ground,
        "camera_matrix": camera_matrix,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "metadata": metadata,
        "keypts_2d": keypts_2d,
        "keypts_3d": keypts_3d,
        "smpl": smpl,
        "ball_positions_2d_only_tracknet": ball_positions_2d_only_tracknet,
        "frames": frames,
        "only_tracknet_second_clipped_slice": only_tracknet_second_clipped_slice,
        # Thinning (indices are relative to video_second_clipped_only_tracknet)
        "dropped_frames": frames_to_drop,
        "video_fps": video_fps,
        "frozen_run": frozen_run,
    }

    return ret


def transform_4dh_to_world_frame(metadata, keypts_2d, keypts_3d, smpl, rvecs, H_ground, scale_factor):
    """
    First, rotates the coordinates into the world frame.
     Next, translates the coordinates under the assumption that one of the feet is on the ground. Here, we use the homography from image plane to ground plane.
    """

    keypts_3d_trans = np.zeros_like(keypts_3d)
    smpl_trans = []

    curr_id = 0
    frame_no = 0
    while curr_id < len(metadata):  # inclusive I think?
        while curr_id < len(metadata) and int(metadata[curr_id]['frame']) < frame_no:
            curr_id += 1
        while curr_id < len(metadata) and int(metadata[curr_id]['frame']) == frame_no:
            keypts = keypts_3d[curr_id]
            s = dict(smpl[curr_id])

            # apply rotation
            rot_mat = cv2.Rodrigues(rvecs[0])[0]
            keypts_rot = (
                keypts @ rot_mat
            )  # right multiplying by R_cw is the same as left multiplying by R_wc. so, we're converting from camera coordinates to world coordinates
            s["global_orient"] = (rot_mat.T @ s["global_orient"][0])[
                None, :, :
            ]  # apply the rotation while preserving the (1, 3, 3) shape

            # apply translation
            min_i = np.argmin(keypts_rot[:, 2])
            _2d_pt = keypts_2d[curr_id][min_i] / scale_factor
            ground_pt = cv2.perspectiveTransform(_2d_pt[None, None, :], np.linalg.inv(H_ground))[0]
            ground_pt_3d = np.array([ground_pt[0, 0], ground_pt[0, 1], 0.0])
            translation = ground_pt_3d[None, :] - keypts_rot[min_i : min_i + 1]
            keypts_rot = keypts_rot + translation
            s["transl"] = translation.flatten()

            keypts_3d_trans[curr_id, :, :] = keypts_rot
            smpl_trans.append(s)
            curr_id += 1
        frame_no += 1

    return keypts_3d_trans, smpl_trans


def _player_features(metadata, smpl):
    """
    For every tracked body near the table: (object_id, mean distance from the table centre, mean translation).

    Also returns how many bodies were tracked in total. Bodies far outside the play area (umpire, spectators)
    are left out.
    """
    player_translations = defaultdict(list)

    curr_id = 0
    frame_no = 0
    while curr_id < len(metadata):  # inclusive I think?
        while curr_id < len(metadata) and int(metadata[curr_id]['frame']) < frame_no:
            curr_id += 1
        while curr_id < len(metadata) and int(metadata[curr_id]['frame']) == frame_no:
            pid = metadata[curr_id]["object_id"]
            s = dict(smpl[curr_id])
            player_translations[pid].append(s["transl"])
            curr_id += 1
        frame_no += 1

    features = []
    for pid, transl in player_translations.items():
        avg_transl = np.mean(np.array(transl), axis=0)
        dists = [np.linalg.norm(transl) for transl in transl]
        avg_dist = np.mean(dists)

        # filter out bodies that are sitting where we'd expect the referee to be sitting or other players walking around
        if np.abs(avg_transl[0]) > 3 * TABLE_LENGTH or np.abs(avg_transl[1]) > TABLE_WIDTH:
            continue

        features.append((pid, avg_dist, avg_transl))

    return features, len(player_translations)


def get_left_right_players(metadata, smpl, keypts_2d):
    """
    Pick the two tracked bodies closest to the table as the left (x < 0) and right players.

    Returns (left_smpl, right_smpl, left_keypts_2d, right_keypts_2d, failure_reason); the reason is "" on success.
    """
    features, num_detected = _player_features(metadata, smpl)

    if len(features) < 2:
        return (
            None,
            None,
            None,
            None,
            f"Less than two total players detected. Originally detected {num_detected} players.",
        )

    # pick two closest players to table
    features_sorted = sorted(features, key=lambda x: x[1])  # sort by avg_dist
    (pid_1, _, avg_transl_1), (pid_2, _, avg_transl_2) = features_sorted[:2]

    if (np.abs(avg_transl_1[0]) < 0.5 or np.abs(avg_transl_2[0]) < 0.5) or np.sign(avg_transl_1[0]) == np.sign(
        avg_transl_2[0]
    ):
        print("Bonkers translation")
        print(avg_transl_1, avg_transl_2)
        return None, None, None, None, "Bonkers translation."

    # the left player is the negative player
    if avg_transl_1[0] < avg_transl_2[0]:
        left_pid, right_pid = pid_1, pid_2
    else:
        left_pid, right_pid = pid_2, pid_1

    # Known limitation: detections are paired by order, not by frame number, and gaps are not infilled.
    left_player_smpl, right_player_smpl = [], []
    left_player_keypts_2d, right_player_keypts_2d = [], []
    for m, s, k_2d in zip(metadata, smpl, keypts_2d):
        pid = m["object_id"]
        if pid == left_pid:
            left_player_smpl.append(s)
            left_player_keypts_2d.append(k_2d)
        elif pid == right_pid:
            right_player_smpl.append(s)
            right_player_keypts_2d.append(k_2d)

    if len(left_player_smpl) != len(right_player_smpl):
        print("Not of equal length!", len(left_player_smpl), len(right_player_smpl))

        # Known limitation: this assumes a player goes missing at the end of the clip, not in the middle.
        if abs(len(left_player_smpl) - len(right_player_smpl)) > 10:
            return (
                None,
                None,
                None,
                None,
                f"SMPL sequences has > {abs(len(left_player_smpl) - len(right_player_smpl)) // 10 * 10} frames in difference.",
            )

        min_length = min(len(left_player_smpl), len(right_player_smpl))
        left_player_smpl = left_player_smpl[:min_length]
        right_player_smpl = right_player_smpl[:min_length]
        left_player_keypts_2d = left_player_keypts_2d[:min_length]
        right_player_keypts_2d = right_player_keypts_2d[:min_length]

    left_player_keypts_2d, right_player_keypts_2d = np.array(left_player_keypts_2d), np.array(right_player_keypts_2d)
    return left_player_smpl, right_player_smpl, left_player_keypts_2d, right_player_keypts_2d, ""


def get_four_players(metadata, smpl, keypts_2d):
    """
    Pick the four tracked bodies closest to the table for doubles, ordered left, right, left, right.

    Returns (four_player_smpl, four_player_keypts_2d), or (None, None) when the sides are not two and two.
    """
    features, _ = _player_features(metadata, smpl)

    if len(features) < 4:
        return None, None

    # pick four closest players to table
    features_sorted = sorted(features, key=lambda x: x[1])  # sort by avg_dist
    four = features_sorted[:4]
    left = [f for f in four if f[2][0] < 0]
    right = [f for f in four if f[2][0] >= 0]
    if len(left) != 2 or len(right) != 2:
        return None, None
    # one player per side first (left, then right) so that skeletons[:2] follows the singles convention
    (pid_1, _, avg_transl_1), (pid_2, _, avg_transl_2), (pid_3, _, avg_transl_3), (pid_4, _, avg_transl_4) = (
        left[0],
        right[0],
        left[1],
        right[1],
    )

    if any([np.abs([avg_transl[0]]) < 0.5 for avg_transl in [avg_transl_1, avg_transl_2, avg_transl_3, avg_transl_4]]):
        print("Bonkers translation")

    four_player_smpl = [[] for _ in range(4)]
    four_player_keypts_2d = [[] for _ in range(4)]
    pid_to_idx = {pid_1: 0, pid_2: 1, pid_3: 2, pid_4: 3}
    for m, s, k_2d in zip(metadata, smpl, keypts_2d):
        pid = m["object_id"]
        if pid in pid_to_idx:
            four_player_smpl[pid_to_idx[pid]].append(s)
            four_player_keypts_2d[pid_to_idx[pid]].append(k_2d)

    if len(set([len(s) for s in four_player_smpl])) != 1:
        print("Not of equal length", [len(s) for s in four_player_smpl])
        # Known limitation: this assumes a player goes missing at the end of the clip, not in the middle.

        min_length = min([len(s) for s in four_player_smpl])
        max_length = max([len(s) for s in four_player_smpl])

        if max_length - min_length > 10:
            return None, None

        for i, s in enumerate(four_player_smpl):
            four_player_smpl[i] = s[:min_length]
        for i, k in enumerate(four_player_keypts_2d):
            four_player_keypts_2d[i] = k[:min_length]

    four_player_keypts_2d = [np.array(k) for k in four_player_keypts_2d]
    return four_player_smpl, four_player_keypts_2d
