# Standard library imports
import copy
import csv
import os

# Third-party imports
import cv2
import numpy as np
import scipy
import torch
from scipy.spatial.transform import Rotation as R

# Local/application imports
from rtt_utils.ball import second_clipping_refinement
from rtt_utils.genmo import process_genmo_player
from rtt_utils.humans import get_four_players, get_left_right_players, load_task, thin_frames, thinned_to_video_index
from rtt_utils.paths import is_doubles, is_doubles_humans, is_singles, is_singles_humans
from rtt_utils.skeleton import (
    MOTIVE21_NAMES,
    define_skeleton_relative_to_root,
    geometric_adjustments_smpl_to_motive,
    select_target21_from_smpl24,
    smpl_to_smplh,
)
from rtt_utils.smpl import (
    load_smpl_model,
    smpl_dicts_to_rotmats,
    smpl_forward_pos,
)
from ttg.types.ttg import TTG

# COCO-17 keypoint indices for wrists
_COCO_LEFT_WRIST = 9
_COCO_RIGHT_WRIST = 10


def _determine_handedness(
    masks_npz_path,
    racket_csv_path,
    vitpose_left,
    vitpose_right,
    start_idx,
    end_idx,
):
    """
    Determine each player's handedness by checking which wrist overlaps
    with detected racket masks across frames.

    Uses vitpose (COCO-17) wrist keypoints and SAM3 racket segmentation masks.
    Votes are accumulated over all frames with detections; the majority-vote
    hand becomes the player's handedness.

    Args:
        masks_npz_path: Path to the racket masks NPZ file.
        racket_csv_path: Path to the racket detections CSV.
        vitpose_left: (N, 17, 3) vitpose keypoints for the left (negative-x) player.
        vitpose_right: (N, 17, 3) vitpose keypoints for the right (positive-x) player.
        start_idx, end_idx: Frame range to clip to.

    Returns:
        List of two strings, e.g. ["right", "left"], indexed same as _humans.
    """
    # Parse racket CSV to get per-frame detections
    frame_detections = {}  # frame_idx -> list of (x1, y1, x2, y2)
    with open(racket_csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            frame = int(row[0])
            if frame < start_idx or frame >= end_idx:
                continue
            if row[2] == '':  # no detection this frame
                continue
            x1, y1, x2, y2 = float(row[4]), float(row[5]), float(row[6]), float(row[7])
            frame_detections.setdefault(frame, []).append((x1, y1, x2, y2))

    # Load masks if available
    masks_data = None
    if os.path.exists(masks_npz_path):
        masks_data = np.load(masks_npz_path)

    # For each player, count votes: how many frames have the racket closer to
    # their left wrist vs right wrist.
    # Index 0 = left player (negative-x), index 1 = right player (positive-x)
    # votes[player_idx] = {"left": count, "right": count}
    votes = [{"left": 0, "right": 0}, {"left": 0, "right": 0}]

    vitpose_by_player = [vitpose_left, vitpose_right]

    for frame, dets in frame_detections.items():
        for det_idx, (x1, y1, x2, y2) in enumerate(dets):
            mask_key = f"frame_{frame}_det_{det_idx}"
            mask = None
            if masks_data is not None and mask_key in masks_data:
                mask = masks_data[mask_key].squeeze()  # (H, W) boolean

            # Find which player's wrist is closest / overlaps this racket
            best_player = None
            best_hand = None
            best_score = -1  # higher is better

            for player_idx, vp in enumerate(vitpose_by_player):
                if frame >= len(vp):
                    continue
                kp = vp[frame]  # (17, 3) — x, y, confidence
                for hand_name, wrist_idx in [("left", _COCO_LEFT_WRIST), ("right", _COCO_RIGHT_WRIST)]:
                    wx, wy, wconf = kp[wrist_idx]
                    if wconf < 0.3:
                        continue

                    if mask is not None:
                        # Check if wrist pixel falls inside the mask
                        px, py = int(round(wx)), int(round(wy))
                        H, W = mask.shape
                        if 0 <= py < H and 0 <= px < W and mask[py, px]:
                            # Wrist is inside the racket mask — strong signal
                            score = 2.0 + wconf
                        else:
                            # Fall back to distance from wrist to bbox center
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                            dist = np.sqrt((wx - cx) ** 2 + (wy - cy) ** 2)
                            score = 1.0 / (1.0 + dist)
                    else:
                        # No mask available, use bbox center distance
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        dist = np.sqrt((wx - cx) ** 2 + (wy - cy) ** 2)
                        score = 1.0 / (1.0 + dist)

                    if score > best_score:
                        best_score = score
                        best_player = player_idx
                        best_hand = hand_name

            if best_player is not None:
                votes[best_player][best_hand] += 1

    # Determine handedness per player from majority vote
    handedness = []
    for player_idx in range(2):
        v = votes[player_idx]
        if v["left"] + v["right"] == 0:
            # No racket detections associated with this player — default to right
            handedness.append("right")
        elif v["right"] >= v["left"]:
            handedness.append("right")
        else:
            handedness.append("left")

    return handedness


class TT4D(TTG):
    def __init__(
        self,
        ball_3d,
        spin_3d,
        skeletons,
        skeletons_2d,
        fps,
        camera_matrix,
        rvecs,
        tvecs,
        H_ground,
        clip_name,
        label,
        ball_2d,
        humans,
        ball_suffix="nointerp",
        gameplay_dir=None,
    ):
        super().__init__(ball_3d, skeletons, fps)
        self.ball_suffix = ball_suffix
        self.spin_3d = spin_3d
        self.skeletons_2d = skeletons_2d  # [(N, 44, 2), (N, 44, 2)]
        self.camera_matrix = camera_matrix
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.H_ground = H_ground
        self.fps = fps
        self.clip_name = clip_name
        self.label = label
        self.ball_2d = ball_2d
        self._humans = humans  # ttg.humans[0][0].keys() dict_keys(['global_orient', 'body_pose', 'betas', 'transl'])
        self.invalid = False
        self.failure_reason = None
        self.segments = None
        self._smplh_humans = None
        self._processed_humans = None

        self.gameplay_dir = gameplay_dir
        self.indices_relative_to_video_second_clipped_only_tracknet = None
        self.processed = False

        # Get camera-to-world transformation
        rvec = np.asarray(rvecs[0]).reshape(3)
        tvec = np.asarray(tvecs[0]).reshape(3)
        R_cw, _ = cv2.Rodrigues(rvec)  # World-to-camera rotation
        R_wc = R_cw.T  # Camera-to-world rotation
        cam_center = -R_cw.T @ tvec  # Camera center in world coordinates
        g_wc = np.eye(4)
        g_wc[:3, :3] = R_wc
        g_wc[:3, 3] = cam_center
        self.g_wc = g_wc

    @classmethod
    def failed(cls, failure_reason):
        """Create a TT4D object representing a failed reconstruction."""
        instance = cls.__new__(cls)
        instance.invalid = True
        instance.failure_reason = failure_reason
        # Set all other attributes to None
        instance.ball_3d = None
        instance.spin_3d = None
        instance._skeletons = None
        instance.skeletons_2d = None
        instance.fps = None
        instance.camera_matrix = None
        instance.rvecs = None
        instance.tvecs = None
        instance.H_ground = None
        instance.clip_name = None
        instance.label = None
        instance.ball_2d = None
        instance._humans = None
        instance.estimated_paddles = None
        instance.segments = None
        instance._smplh_humans = None
        instance._processed_humans = None

        instance.gameplay_dir = None
        instance.ball_suffix = None
        instance.indices_relative_to_video_second_clipped_only_tracknet = None
        instance.processed = False

        instance.g_wc = None
        instance.handedness = None
        return instance

    @property
    def humans(self):
        if hasattr(self, "pose_source") and self.pose_source == "genmo":
            return self._humans
        # if self._processed_humans is not None:
        #     return self._processed_humans

        num_humans = len(self._humans)
        human_seqs = [copy.copy(seq) for seq in self._humans]
        T = len(human_seqs[0])

        # Extract translation arrays for each human: list of (T, 3) arrays
        transls = []
        for seq in human_seqs:
            transl = np.concatenate([s["transl"].reshape(1, 3) for s in seq], axis=0)
            transls.append(transl)

        # --- Per-human consistency check: x-sign must not change over time ---
        for idx, transl in enumerate(transls):
            has_neg = (transl[:, 0] < 0).any()
            has_pos = (transl[:, 0] > 0).any()
            if has_neg and has_pos:
                print(f"Inconsistent x-sign for human {idx} across time.")
                self.invalid = True
                self.failure_reason = f"Inconsistent x-sign for human {idx} across time."
                return

        # --- Side assignment check ---
        # Determine which side each human is on based on the sign of their x-translation
        left_indices = [i for i, t in enumerate(transls) if (t[:, 0] < 0).all()]
        right_indices = [i for i, t in enumerate(transls) if (t[:, 0] > 0).all()]

        if len(left_indices) + len(right_indices) != num_humans:
            self.invalid = True
            self.failure_reason = "Some humans have all-zero x-translations; cannot determine side."
            return

        if num_humans == 2:
            # Singles: exactly one on each side
            if len(left_indices) != 1 or len(right_indices) != 1:
                self.invalid = True
                self.failure_reason = (
                    f"Improper assignment of left and right poses "
                    f"(left={len(left_indices)}, right={len(right_indices)})"
                )
                raise Exception(self.failure_reason)
        elif num_humans == 4:
            # Doubles: exactly two on each side
            if len(left_indices) != 2 or len(right_indices) != 2:
                self.invalid = True
                self.failure_reason = (
                    f"Doubles requires exactly 2 players per side "
                    f"(left={len(left_indices)}, right={len(right_indices)})"
                )
                raise Exception(self.failure_reason)

        # --- Box-filter smoothing on translations ---
        # filter ... homography makes things noisy. we saw this with normal vector estimation of racket
        if T > 10:
            w = np.array([1, 1, 1, 1], dtype=np.float32) / 4.0  # box filter
            for transl in transls:
                for ax in range(3):
                    padded = np.pad(
                        transl[:, ax],
                        ((len(w) - 1) // 2, len(w) // 2),
                        mode="edge",
                    )
                    transl[:, ax] = np.convolve(padded, w, mode="valid")

        # Write smoothed translations back into the per-frame dicts
        for seq, transl in zip(human_seqs, transls):
            for i, s in enumerate(seq):
                s["transl"] = transl[i].reshape((3,))

        # Store with canonical ordering: all left-side humans first, then right
        self._processed_humans = [human_seqs[i] for i in left_indices] + [human_seqs[i] for i in right_indices]

        return self._processed_humans

    @property
    def smplh_humans(self):
        """
        Convert SMPL humans data to SMPLH 52-joint positions for retargeting.

        Returns:
            List of two numpy arrays [T, 52, 3], one for each player.
            The arrays contain joint positions in SMPLH format, suitable for
            use with holosoma_retargeting.
        """
        # if False and self._smplh_humans is not None:
        #     return self._smplh_humans

        humans = self.humans
        if humans is None:
            return

        # SMPLH leg joint indices:
        # Left leg:  L_Hip(1), L_Knee(2), L_Ankle(3), L_Toe(4)
        # Right leg: R_Hip(5), R_Knee(6), R_Ankle(7), R_Toe(8)
        L_TOE_IDX = 4
        R_TOE_IDX = 8

        # SMPLH hip joint indices:
        # Left hip: L_Hip(1)
        # Right hip: R_Hip(5)

        self._smplh_humans = []
        for human in humans:
            human_skeleton = smpl_to_smplh(human)  # Player i: [T, 52, 3]

            human_skeleton[:, L_TOE_IDX, 2]
            human_skeleton[:, R_TOE_IDX, 2]

            # # want the toe_z to be at the same offset
            # # this method will cause the humanoid to crouch more. an alternative is to move the higher joint to the lower joint
            # # this also destroys the gait
            # toe_offset = left_toe_z - right_toe_z
            # human_skeleton[:, LEFT_LEG_JOINTS, 2] -= (toe_offset / 2).reshape(-1, 1)
            # human_skeleton[:, RIGHT_LEG_JOINTS, 2] += (toe_offset / 2).reshape(-1, 1)

            # toe_height = human_skeleton[:, L_TOE_IDX, 2].copy() # only have to look at the left toe because they share the same z now at every frame
            # human_skeleton[:, LEFT_LEG_JOINTS, 2] -= toe_height.reshape(-1, 1)
            # human_skeleton[:, RIGHT_LEG_JOINTS, 2] -= toe_height.reshape(-1, 1)

            # the below hovers too much!!!

            # left_toe_z = np.median(human_skeleton[:, L_TOE_IDX, 2])
            # right_toe_z = np.median(human_skeleton[:, R_TOE_IDX, 2])

            # # want the toe_z to be at the same offset
            # # this method will cause the humanoid to crouch more. an alternative is to move the higher joint to the lower joint
            # # this method maintains the gait better
            # toe_offset = left_toe_z - right_toe_z
            # human_skeleton[:, LEFT_LEG_JOINTS, 2] -= toe_offset / 2
            # human_skeleton[:, RIGHT_LEG_JOINTS, 2] += toe_offset / 2

            # human_skeleton[:, LEFT_LEG_JOINTS, 2] -= np.median(human_skeleton[:, L_TOE_IDX, 2])
            # human_skeleton[:, RIGHT_LEG_JOINTS, 2] -= np.median(human_skeleton[:, R_TOE_IDX, 2])

            # # nudge the hips so that they are more leveled
            # dev = human_skeleton[:, LEFT_HIP_IDX, 2] - human_skeleton[:, RIGHT_HIP_IDX, 2]
            # dev /= 2 # bring it closer to zero so that the hips are less slanted
            # human_skeleton[:, LEFT_HIP_IDX, 2] = human_skeleton[:, RIGHT_HIP_IDX, 2] + dev

            self._smplh_humans.append(human_skeleton)

        return self._smplh_humans

    def hit_point_identification(self):
        if self.segments is not None:
            return

        # Pass ball_3d to second_clipping_refinement via dict (function expects dict with "ball_3d" key)
        ret = second_clipping_refinement({"ball_3d": self.ball_3d}, gameplay_dir=self.gameplay_dir)
        if ret is None:
            self.invalid = True
            self.failure_reason = "Second clipping refinement failure"
            return

        _, _, hit_indices = ret
        # self.second_clipped_refinement = (int(hit_indices[0]), int(hit_indices[-1]))
        self.set_hit_indices(hit_indices)

    def slice_gameplay(self, start_idx=0, end_idx=-1):
        # end_idx is Python-exclusive: ball_3d[start_idx:end_idx] gives end_idx - start_idx
        # frames. To keep a hit at index K as the last frame, callers pass end_idx = K + 1.
        if end_idx == -1:
            end_idx = len(self.ball_3d)

        # assert within acceptable range
        assert (start_idx >= 0) and ((end_idx - 1) <= self.segments[-1][1])

        # Slice all main features to new range
        self.ball_2d = self.ball_2d[start_idx:end_idx]
        self.ball_3d = self.ball_3d[start_idx:end_idx]
        self.interpolated_ball_3d = self.interpolated_ball_3d[start_idx:end_idx]
        self.spin_3d = self.spin_3d[start_idx:end_idx]
        self._skeletons = [sk[start_idx:end_idx] for sk in self._skeletons]
        self.skeletons_2d = [sk[start_idx:end_idx] for sk in self.skeletons_2d]
        self._humans = [h[start_idx:end_idx] for h in self._humans]

        # Adjust segments and hit_indices to be relative to the sliced data.
        # Filter is `< end_idx` to match Python-exclusive slicing — a hit at end_idx
        # is past the last array frame.
        self.segments = [
            (s0 - start_idx, s1 - start_idx) for (s0, s1) in self.segments if (s0 - start_idx >= 0) and (s1 < end_idx)
        ]
        self.hit_indices = [h - start_idx for h in self.hit_indices if (h - start_idx >= 0) and (h < end_idx)]
        self.exchanges = [
            (a - start_idx, b - start_idx) for a, b in self.exchanges if (a - start_idx >= 0) and (b < end_idx)
        ]

        if self.indices_relative_to_video_second_clipped_only_tracknet is None:
            self.indices_relative_to_video_second_clipped_only_tracknet = (start_idx, end_idx)
        else:
            a, _ = self.indices_relative_to_video_second_clipped_only_tracknet
            self.indices_relative_to_video_second_clipped_only_tracknet = (a + start_idx, a + end_idx)

    def copy_slice_gameplay(self, start=0, end=-1):
        new = copy.copy(self)
        new.slice_gameplay(start, end)
        return new

    def copy_segment_gameplay(self, idx):
        new = copy.copy(self)
        s0, s1 = new.segments[idx]
        new.slice_gameplay(s0, s1 + 1)
        return new

    def copy_exchange_gameplay(self, exchange_index):
        assert exchange_index < len(self.exchanges)
        start_idx, end_hit = self.exchanges[exchange_index]
        end_idx = end_hit + 1
        new = self.copy_slice_gameplay(start_idx, end_idx)

        initial_ball_states = []
        for i, (s0, s1) in enumerate(self.segments):
            if (s0 - start_idx >= 0) and (s1 < end_idx):
                initial_ball_states.append(self.initial_ball_states[i])
        new.initial_ball_states = initial_ball_states
        assert len(new.initial_ball_states) == 2

        return new

    def process(self, smpl_model=None):
        if self.processed:
            return

        self.hit_point_identification()
        if self.invalid:
            return

        # refine the start and end of the data
        start_idx = self.segments[0][0]
        end_idx = self.segments[-1][1] + 1  # slicing is exclusive

        self.annotate_ode_fit()
        self.annotate_reprojection_errors()
        # self.add_racket_annotations()

        # if self.label == "singles":
        #     self.translation_fix_wrist_ball(smpl_model)

        self.slice_gameplay(start_idx, end_idx)
        self.processed = True

    # SMPL joint indices for wrists
    _SMPL_LEFT_WRIST = 20
    _SMPL_RIGHT_WRIST = 21

    def translation_fix_wrist_ball(self, smpl_model=None, max_offset=0.5, intensity=0.75):
        """
        Adjust each player's x-y translation so that at hit times, the hitting
        wrist is closer to the ball. Offsets are computed at hit frames, thresholded,
        and interpolated smoothly across all frames per player.
        """
        if smpl_model is None:
            smpl_model = self._get_smpl()

        # Determine hitting wrist joint index per player
        wrist_joint = []
        for h in self.handedness:
            if h == "right":
                wrist_joint.append(self._SMPL_RIGHT_WRIST)
            else:
                wrist_joint.append(self._SMPL_LEFT_WRIST)

        # Run SMPL forward pass per player to get joint positions
        pos24_per_player = []
        for player_idx in range(2):
            go, bp, tr, bt = smpl_dicts_to_rotmats(self._humans[player_idx])
            pos24, _ = smpl_forward_pos(smpl_model, go, bp, tr, bt)
            pos24_per_player.append(pos24)

        ball_3d = self.interpolated_ball_3d.copy()
        # annotate_ode_fit fills interpolated_ball_3d[start:end] (exclusive end), so
        # the very last segment's endpoint is NaN. Copy from the previous frame.
        for _seg_start, seg_end in self.segments:
            if np.any(np.isnan(ball_3d[seg_end])):
                ball_3d[seg_end] = ball_3d[seg_end - 1]

        # Collect per-player sparse offsets: (frame_index, xy_offset)
        per_player_hits = {0: [], 1: []}
        for hit_idx in self.hit_indices:
            ball_x = ball_3d[hit_idx, 0]
            player_idx = 0 if ball_x < 0 else 1

            wrist_pos = pos24_per_player[player_idx][hit_idx, wrist_joint[player_idx]]
            offset_xy = ball_3d[hit_idx, :2] - wrist_pos[:2]

            # Clamp each axis independently to max_offset
            offset_xy = np.clip(offset_xy, -max_offset, max_offset)
            per_player_hits[player_idx].append((hit_idx, offset_xy * intensity))

        # Interpolate and apply per player
        T = len(self._humans[0])
        for player_idx in range(2):
            hits = per_player_hits[player_idx]
            if len(hits) < 2:
                continue

            indices = np.array([h[0] for h in hits])
            offsets = np.array([h[1] for h in hits])  # (N, 2)

            interp_x = scipy.interpolate.PchipInterpolator(indices, offsets[:, 0])
            interp_y = scipy.interpolate.PchipInterpolator(indices, offsets[:, 1])

            all_t = np.arange(T)
            dx = interp_x(all_t)
            dy = interp_y(all_t)

            for t in range(T):
                self._humans[player_idx][t]["transl"][0] += dx[t]
                self._humans[player_idx][t]["transl"][1] += dy[t]

            # Translate the skeleton so its hitting-wrist xy matches the
            # post-fix SMPL wrist xy (pos24 + dx/dy).
            if self._skeletons is not None:
                smpl_wrist_idx = wrist_joint[player_idx]
                motive_wrist_idx = 12 if self.handedness[player_idx] == "right" else 8

                target_wrist_xy = pos24_per_player[player_idx][:, smpl_wrist_idx, :2].copy()
                target_wrist_xy[:, 0] += dx
                target_wrist_xy[:, 1] += dy

                current_wrist_xy = self._skeletons[player_idx][:, motive_wrist_idx, :2]
                offset_xy = target_wrist_xy - current_wrist_xy  # (T, 2)

                self._skeletons[player_idx][:, :, 0] += offset_xy[:, 0:1]
                self._skeletons[player_idx][:, :, 1] += offset_xy[:, 1:2]

    def annotate_reprojection_errors(self):
        rvec = np.asarray(self.rvecs[0]).reshape(3)
        tvec = np.asarray(self.tvecs[0]).reshape(3, 1)
        R_cw, _ = cv2.Rodrigues(rvec)

        self.reprojection_errors_neural = []
        self.reprojection_errors_interpolation = []
        for start, end in self.segments:
            ball_3d = self.ball_3d[start:end]
            ball_2d = self.ball_2d[start:end]

            vis = ball_2d[:, 1].astype(bool)
            ball_2d_coords = ball_2d[:, 2:4]

            pixel_projection = (self.camera_matrix @ (R_cw @ ball_3d.T + tvec)).T
            pixel_projection = pixel_projection / pixel_projection[:, 2:3]
            pixel_projection = pixel_projection[:, :2]
            reprojection_error = np.abs(pixel_projection - ball_2d_coords)[vis]
            self.reprojection_errors_neural.append(reprojection_error)

            ball_3d = self.interpolated_ball_3d[start:end]
            pixel_projection = (self.camera_matrix @ (R_cw @ ball_3d.T + tvec)).T
            pixel_projection = pixel_projection / pixel_projection[:, 2:3]
            pixel_projection = pixel_projection[:, :2]
            reprojection_error = np.abs(pixel_projection - ball_2d_coords)[vis]
            self.reprojection_errors_interpolation.append(reprojection_error)

    @classmethod
    def _get_smpl(cls, device=None):
        """Load SMPL model using smplx library."""
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        smpl_model = load_smpl_model(device=device)
        return smpl_model

    @staticmethod
    def _load_ball_spin(gameplay_dir, ball_positions_2d_only_tracknet, ball_suffix="nointerp"):
        """Load uplifted 3D ball + spin, scattering visible predictions into full-length arrays.

        Returns:
            (ball_3d, spin_3d, None) on success, or (None, None, error_msg) on failure.
        """
        # N = ball_positions_2d_only_tracknet.shape[0]
        # ball_3d = np.zeros((N, 3), dtype=float)
        # spin_3d = np.zeros((N, 3), dtype=float)
        # return ball_3d, spin_3d, None

        # temporary
        # ball_dir = os.path.join(gameplay_dir, "dani_ball")
        # ball_filename = f"3d_ball.npy"
        # spin_filename = f"3d_spin.npy"
        # ball_path = os.path.join(ball_dir, ball_filename)
        # spin_path = os.path.join(ball_dir, spin_filename)
        # ball_3d = np.load(ball_path)
        # spin_3d = np.load(spin_path)
        # return ball_3d, spin_3d, None

        ball_dir = os.path.join(gameplay_dir, "dani_ball")
        ball_filename = f"3d_ball_{ball_suffix}.npy"
        spin_filename = f"3d_spin_{ball_suffix}.npy"
        ball_path = os.path.join(ball_dir, ball_filename)
        spin_path = os.path.join(ball_dir, spin_filename)
        for p, label in [(ball_path, ball_filename), (spin_path, spin_filename)]:
            if not os.path.exists(p):
                return None, None, f"No {label} found"

        N = ball_positions_2d_only_tracknet.shape[0]
        ball_3d = np.zeros((N, 3), dtype=float)
        spin_3d = np.zeros((N, 3), dtype=float)
        if ball_suffix == "nointerp":
            vis = ball_positions_2d_only_tracknet[:, 1].astype(bool)
            ball_3d[vis] = np.load(ball_path)[vis]
            spin_3d[vis] = np.load(spin_path)[vis]
        else:
            ball_3d = np.load(ball_path)
            spin_3d = np.load(spin_path)
        return ball_3d, spin_3d, None

    @classmethod
    def _from_gameplay_dir_singles_genmo(
        cls, gameplay_dir, smpl_model=None, ball_suffix="nointerp", load_from="processed", needs_spark_result=False
    ):
        """
        Factory method to create a TT4D object from a singles gameplay directory
        using GENMO output for human pose estimation for both players.

        Args:
            gameplay_dir: Path to the gameplay directory
            smpl_model: Optional pre-loaded SMPL model (for efficiency when processing multiple clips)

        Returns:
            TT4D object (may be a failed instance if data is missing or invalid)
        """

        def _genmo_path(subdir):
            return os.path.join(gameplay_dir, "human_pose_tracker_only_tracknet", subdir)

        genmo_results_path = _genmo_path("GENMO/hmr4d_results.pt")
        genmo_positive_results_path = _genmo_path("GENMO_positive_player/hmr4d_results.pt")
        vitpose_path = _genmo_path("GENMO/preprocess/vitpose.pt")
        vitpose_positive_path = _genmo_path("GENMO_positive_player/preprocess/vitpose.pt")

        for path, label in [
            (genmo_results_path, "GENMO results"),
            (genmo_positive_results_path, "GENMO positive player results"),
            (vitpose_path, "GENMO vitpose.pt"),
            (vitpose_positive_path, "GENMO positive player vitpose.pt"),
        ]:
            if not os.path.exists(path):
                return cls.failed(f"{label} not found at {path}")

        clip_name = gameplay_dir.split("/")[-1]
        if load_from == "processed":
            pkl_path = os.path.join(gameplay_dir, "nima_recons", f"{clip_name}_tt4d_processed.pkl")
        elif load_from == "partial":
            pkl_path = os.path.join(gameplay_dir, "nima_recons", f"{clip_name}_tt4d.pkl")
        else:
            raise Exception(f"Not a valid input for load_from: {load_from}")
        if not os.path.exists(pkl_path):
            return cls.failed(f"No existing TT4D result found at {pkl_path}")

        if smpl_model is None:
            smpl_model = cls._get_smpl()

        tt4d = TT4D.load(pkl_path)
        tt4d.pose_source = "genmo"

        def _load_genmo(path, label):
            try:
                pred = torch.load(path, map_location="cpu")
            except Exception as e:
                raise ValueError(f"Failed to load {label}: {e}")
            if "smpl_params_global" not in pred:
                raise ValueError(f"{label} missing 'smpl_params_global'")
            return pred

        def _load_vitpose(path):
            vp = torch.load(path, map_location="cpu")
            if isinstance(vp, tuple):
                vp = vp[0]
            return vp.numpy() if isinstance(vp, torch.Tensor) else vp

        ret = load_task(gameplay_dir, thin=True)
        H_ground = ret["H_ground"]

        try:
            genmo_pred = _load_genmo(genmo_results_path, "GENMO results")
            genmo_pred_right = _load_genmo(genmo_positive_results_path, "GENMO positive player results")
            vitpose = _load_vitpose(vitpose_path)
            vitpose_right = _load_vitpose(vitpose_positive_path)

            left_player_smpl, _, left_foot_contact = process_genmo_player(
                genmo_pred, tt4d.g_wc, smpl_model, vitpose, H_ground
            )
            right_player_smpl, _, right_foot_contact = process_genmo_player(
                genmo_pred_right, tt4d.g_wc, smpl_model, vitpose_right, H_ground
            )
        except ValueError as e:
            return cls.failed(str(e))

        # The pkl was built on the thinned frames (load_task); thin the per-frame GENMO outputs the same way.
        dropped_frames = getattr(tt4d, "dropped_frames", None)
        num_video_frames = len(left_player_smpl)
        left_player_smpl, right_player_smpl, left_foot_contact, right_foot_contact = [
            thin_frames(x, dropped_frames)
            for x in (left_player_smpl, right_player_smpl, left_foot_contact, right_foot_contact)
        ]

        if tt4d.indices_relative_to_video_second_clipped_only_tracknet is None:
            return cls.failed("Ugh indices_relative_to_video_second_clipped_only_tracknet is None??")
        start_idx, end_idx = tt4d.indices_relative_to_video_second_clipped_only_tracknet

        tt4d._humans = [
            left_player_smpl[start_idx:end_idx],
            right_player_smpl[start_idx:end_idx],
        ]

        skeleton_save_dir = os.path.join(gameplay_dir, "cory337")
        negative_skeleton_path = os.path.join(skeleton_save_dir, "negative_player.npz")
        positive_skeleton_path = os.path.join(skeleton_save_dir, "positive_player.npz")

        if os.path.exists(negative_skeleton_path) and os.path.exists(positive_skeleton_path):
            neg_data = np.load(negative_skeleton_path)
            pos_data = np.load(positive_skeleton_path)
            neg_pos = thin_frames(neg_data["retarget_pos"], dropped_frames)
            pos_pos = thin_frames(pos_data["retarget_pos"], dropped_frames)
            # Reorder from cory337 depth-first order to MOTIVE21 order
            cory337_names = list(neg_data["retarget_joint_names"])
            reorder = [cory337_names.index(name) for name in MOTIVE21_NAMES]
            tt4d._skeletons = [
                neg_pos[start_idx:end_idx][:, reorder],
                pos_pos[start_idx:end_idx][:, reorder],
            ]
        else:
            if needs_spark_result:
                return cls.failed("No spark result found.")

        tt4d.foot_contact = [
            left_foot_contact[start_idx:end_idx],
            right_foot_contact[start_idx:end_idx],
        ]

        racket_track_file = os.path.join(gameplay_dir, f"racket_tracker_sam3/{clip_name}_racket.csv")
        racket_masks_file = os.path.join(gameplay_dir, f"racket_tracker_sam3/{clip_name}_masks.npz")

        if os.path.exists(racket_track_file):
            # racket detections and vitpose are indexed by video frame, so map the thinned window back
            tt4d.handedness = _determine_handedness(
                racket_masks_file,
                racket_track_file,
                vitpose,
                vitpose_right,
                thinned_to_video_index(start_idx, dropped_frames, num_video_frames),
                thinned_to_video_index(end_idx - 1, dropped_frames, num_video_frames) + 1,
            )
        else:
            return cls.failed("No racket detection found.")

        tt4d.gameplay_dir = gameplay_dir
        return tt4d

    @classmethod
    def _from_gameplay_dir_singles_genmo_full_clip(
        cls, gameplay_dir, smpl_model=None, ball_suffix="nointerp", load_from="processed", needs_spark_result=False
    ):
        """
        Factory method to create a TT4D object from a singles gameplay directory
        using GENMO output for human pose estimation for both players.

        Args:
            gameplay_dir: Path to the gameplay directory
            smpl_model: Optional pre-loaded SMPL model (for efficiency when processing multiple clips)

        Returns:
            TT4D object (may be a failed instance if data is missing or invalid)
        """

        def _genmo_path(subdir):
            return os.path.join(gameplay_dir, "human_pose_tracker_only_tracknet", subdir)

        genmo_results_path = _genmo_path("GENMO/hmr4d_results.pt")
        genmo_positive_results_path = _genmo_path("GENMO_positive_player/hmr4d_results.pt")
        vitpose_path = _genmo_path("GENMO/preprocess/vitpose.pt")
        vitpose_positive_path = _genmo_path("GENMO_positive_player/preprocess/vitpose.pt")

        for path, label in [
            (genmo_results_path, "GENMO results"),
            (genmo_positive_results_path, "GENMO positive player results"),
            (vitpose_path, "GENMO vitpose.pt"),
            (vitpose_positive_path, "GENMO positive player vitpose.pt"),
        ]:
            if not os.path.exists(path):
                return cls.failed(f"{label} not found at {path}")

        clip_name = gameplay_dir.split("/")[-1]
        if load_from == "processed":
            pkl_path = os.path.join(gameplay_dir, "nima_recons", f"{clip_name}_tt4d_processed.pkl")
        elif load_from == "partial":
            pkl_path = os.path.join(gameplay_dir, "nima_recons", f"{clip_name}_tt4d.pkl")
        else:
            raise Exception(f"Not a valid input for load_from: {load_from}")
        if not os.path.exists(pkl_path):
            return cls.failed(f"No existing TT4D result found at {pkl_path}")

        if smpl_model is None:
            smpl_model = cls._get_smpl()

        tt4d = TT4D.load(pkl_path)
        tt4d.pose_source = "genmo"

        def _load_genmo(path, label):
            try:
                pred = torch.load(path, map_location="cpu")
            except Exception as e:
                raise ValueError(f"Failed to load {label}: {e}")
            if "smpl_params_global" not in pred:
                raise ValueError(f"{label} missing 'smpl_params_global'")
            return pred

        def _load_vitpose(path):
            vp = torch.load(path, map_location="cpu")
            if isinstance(vp, tuple):
                vp = vp[0]
            return vp.numpy() if isinstance(vp, torch.Tensor) else vp

        ret = load_task(gameplay_dir, thin=True)
        H_ground = ret["H_ground"]

        try:
            genmo_pred = _load_genmo(genmo_results_path, "GENMO results")
            genmo_pred_right = _load_genmo(genmo_positive_results_path, "GENMO positive player results")
            vitpose = _load_vitpose(vitpose_path)
            vitpose_right = _load_vitpose(vitpose_positive_path)

            left_player_smpl, _, left_foot_contact = process_genmo_player(
                genmo_pred, tt4d.g_wc, smpl_model, vitpose, H_ground
            )
            right_player_smpl, _, right_foot_contact = process_genmo_player(
                genmo_pred_right, tt4d.g_wc, smpl_model, vitpose_right, H_ground
            )
        except ValueError as e:
            return cls.failed(str(e))

        # The pkl was built on the thinned frames (load_task); thin the per-frame GENMO outputs the same way.
        dropped_frames = getattr(tt4d, "dropped_frames", None)
        num_video_frames = len(left_player_smpl)
        left_player_smpl, right_player_smpl, left_foot_contact, right_foot_contact = [
            thin_frames(x, dropped_frames)
            for x in (left_player_smpl, right_player_smpl, left_foot_contact, right_foot_contact)
        ]

        if tt4d.indices_relative_to_video_second_clipped_only_tracknet:
            return tt4d
        start_idx, end_idx = tt4d.indices_relative_to_video_second_clipped_only_tracknet
        # The pkl stores interpolated_ball_3d / segments / hit_indices in clipped-relative
        # coordinates (slice_gameplay subtracted start_idx). Undo that shift so they line
        # up with the full-clip _humans we are about to install, padding the ball with
        # zeros outside the clipped range.
        T_full = len(left_player_smpl)
        full_ball = np.zeros((T_full, 3), dtype=tt4d.interpolated_ball_3d.dtype)
        full_ball[start_idx:end_idx] = tt4d.interpolated_ball_3d
        tt4d.interpolated_ball_3d = full_ball
        tt4d.segments = [(s0 + start_idx, s1 + start_idx) for (s0, s1) in tt4d.segments]
        tt4d.hit_indices = [h + start_idx for h in tt4d.hit_indices]

        start_idx, end_idx = 0, 10000

        tt4d._humans = [
            left_player_smpl[start_idx:end_idx],
            right_player_smpl[start_idx:end_idx],
        ]

        skeleton_save_dir = os.path.join(gameplay_dir, "cory337")
        negative_skeleton_path = os.path.join(skeleton_save_dir, "negative_player.npz")
        positive_skeleton_path = os.path.join(skeleton_save_dir, "positive_player.npz")

        if os.path.exists(negative_skeleton_path) and os.path.exists(positive_skeleton_path):
            neg_data = np.load(negative_skeleton_path)
            pos_data = np.load(positive_skeleton_path)
            neg_pos = thin_frames(neg_data["retarget_pos"], dropped_frames)
            pos_pos = thin_frames(pos_data["retarget_pos"], dropped_frames)
            # Reorder from cory337 depth-first order to MOTIVE21 order
            cory337_names = list(neg_data["retarget_joint_names"])
            reorder = [cory337_names.index(name) for name in MOTIVE21_NAMES]
            tt4d._skeletons = [
                neg_pos[start_idx:end_idx][:, reorder],
                pos_pos[start_idx:end_idx][:, reorder],
            ]
        else:
            if needs_spark_result:
                return cls.failed("No spark result found.")

        tt4d.foot_contact = [
            left_foot_contact[start_idx:end_idx],
            right_foot_contact[start_idx:end_idx],
        ]

        racket_track_file = os.path.join(gameplay_dir, f"racket_tracker_sam3/{clip_name}_racket.csv")
        racket_masks_file = os.path.join(gameplay_dir, f"racket_tracker_sam3/{clip_name}_masks.npz")

        if os.path.exists(racket_track_file):
            # racket detections and vitpose are indexed by video frame, so map the thinned window back
            tt4d.handedness = _determine_handedness(
                racket_masks_file,
                racket_track_file,
                vitpose,
                vitpose_right,
                thinned_to_video_index(start_idx, dropped_frames, num_video_frames),
                thinned_to_video_index(end_idx - 1, dropped_frames, num_video_frames) + 1,
            )
        else:
            return cls.failed("No racket detection found.")

        tt4d.gameplay_dir = gameplay_dir
        return tt4d

    @classmethod
    def _from_gameplay_dir_singles_partial(cls, gameplay_dir, smpl_model=None, ball_suffix="nointerp"):
        """
        Factory method to create a TT4D object from a singles gameplay directory.

        Args:
            gameplay_dir: Path to the gameplay directory
            smpl_model: Optional pre-loaded SMPL model (for efficiency when processing multiple clips)

        Returns:
            TT4D object (may be a failed instance if data is missing or invalid)
        """
        if smpl_model is None:
            smpl_model = cls._get_smpl()

        clip_name = gameplay_dir.split("/")[-1]

        ret = load_task(gameplay_dir, thin=True)
        (
            camera_matrix,
            rvecs,
            tvecs,
            fps,
            H_ground,
            metadata,
            _keypts_3d,
            keypts_2d,
            smpl,
            ball_positions_2d_only_tracknet,
            _frames,
            _gameplay_slice,
        ) = (
            ret["camera_matrix"],
            ret["rvecs"],
            ret["tvecs"],
            ret["fps"],
            ret["H_ground"],
            ret["metadata"],
            ret["keypts_3d"],
            ret["keypts_2d"],
            ret["smpl"],
            ret["ball_positions_2d_only_tracknet"],
            ret["frames"],
            ret["only_tracknet_second_clipped_slice"],
        )
        left_player_smpl, right_player_smpl, left_player_keypts_2d, right_player_keypts_2d, failure_reason = (
            get_left_right_players(metadata, smpl, keypts_2d)
        )
        if failure_reason != "":
            return cls.failed(f"No SMPL mesh found. {failure_reason}")

        if len(left_player_smpl) == 0:
            return cls.failed("No SMPL mesh found (len 0)")

        # Check for out of sync data
        if abs(len(left_player_smpl) - ball_positions_2d_only_tracknet.shape[0]) > 5:
            return cls.failed(
                f"Data out of sync (likely due to SMPL). Length of mismatch: {abs(len(left_player_smpl) - ball_positions_2d_only_tracknet.shape[0])}"
            )

        # Trim all relevant lists / arrays down to the minimum available length
        # so downstream code can assume consistent per-frame indexing.
        L = min(
            len(left_player_smpl),
            len(right_player_smpl),
            len(left_player_keypts_2d),
            len(right_player_keypts_2d),
            ball_positions_2d_only_tracknet.shape[0],
        )
        left_player_smpl = left_player_smpl[:L]
        right_player_smpl = right_player_smpl[:L]
        left_player_keypts_2d = left_player_keypts_2d[:L]
        right_player_keypts_2d = right_player_keypts_2d[:L]
        ball_positions_2d_only_tracknet = ball_positions_2d_only_tracknet[:L]

        # SMPL to BVH conversion
        go_mats, bp_mats, trans, betas = smpl_dicts_to_rotmats(left_player_smpl)
        pos24, _ = smpl_forward_pos(smpl_model, go_mats, bp_mats, trans, betas)
        left_player_bvh, _ = select_target21_from_smpl24(pos24)

        go_mats, bp_mats, trans, betas = smpl_dicts_to_rotmats(right_player_smpl)
        pos24, _ = smpl_forward_pos(smpl_model, go_mats, bp_mats, trans, betas)
        right_player_bvh, _ = select_target21_from_smpl24(pos24)

        # Create TT4D object
        tt4d = cls(
            ball_3d=None,
            spin_3d=None,
            skeletons=[left_player_bvh, right_player_bvh],
            skeletons_2d=[left_player_keypts_2d, right_player_keypts_2d],
            fps=fps,
            camera_matrix=camera_matrix,
            rvecs=rvecs,
            tvecs=tvecs,
            H_ground=H_ground,
            clip_name=clip_name,
            label="singles",
            ball_2d=ball_positions_2d_only_tracknet,
            humans=[left_player_smpl, right_player_smpl],
            ball_suffix=ball_suffix,
        )

        # Video thinning (see load_task): every per-frame feature above has the duplicated video
        # frames removed and fps is the resulting rate. A long run of duplicates means the video froze.
        if ret["frozen_run"] >= 3:
            return cls.failed(f"Video froze inside the rally ({ret['frozen_run']} consecutive duplicated frames)")
        tt4d.dropped_frames = ret["dropped_frames"]
        tt4d.video_fps = ret["video_fps"]

        tt4d.gameplay_dir = gameplay_dir
        return tt4d

    @classmethod
    def _from_gameplay_dir_singles(cls, gameplay_dir, smpl_model=None, ball_suffix="nointerp"):
        # # Check for existing 4DHumans-based TT4D object
        # clip_name = gameplay_dir.split("/")[-1]
        # pkl_processed_path = os.path.join(gameplay_dir, "nima_recons", f"{clip_name}_tt4d.pkl")
        # if not os.path.exists(pkl_processed_path):
        #     return cls.failed(f"No existing TT4D result found at {pkl_processed_path}")

        # if smpl_model is None:
        #     smpl_model = cls._get_smpl()

        # # Base object
        # tt4d = TT4D.load(pkl_processed_path)

        # ball_3d, spin_3d, failure_reason = TT4D._load_ball_spin(gameplay_dir, tt4d.ball_2d, ball_suffix)
        # if failure_reason is not None:
        #     return cls.failed(f"Failed on _load_ball_spin: {failure_reason}.")
        # tt4d.ball_3d = ball_3d
        # tt4d.spin_3d = spin_3d

        # tt4d.foot_contact = np.zeros((tt4d.skeletons[0].shape[0], 2))

        # Check for existing 4DHumans-based TT4D object
        clip_name = gameplay_dir.split("/")[-1]
        pkl_processed_path = os.path.join(gameplay_dir, "nima_recons", f"{clip_name}_tt4d_processed.pkl")
        if not os.path.exists(pkl_processed_path):
            return cls.failed(f"No existing TT4D result found at {pkl_processed_path}")

        # if smpl_model is None:
        #     smpl_model = cls._get_smpl()

        # Base object
        tt4d = TT4D.load(pkl_processed_path)

        tt4d.gameplay_dir = gameplay_dir
        # tt4d.foot_contact = np.zeros((tt4d.skeletons[0].shape[0], 2))

        return tt4d

    @classmethod
    def _from_gameplay_dir_doubles_partial(cls, gameplay_dir, smpl_model=None, ball_suffix="nointerp"):
        """
        Factory method to create a TT4D object from a doubles gameplay directory.

        Args:
            gameplay_dir: Path to the gameplay directory
            smpl_model: Optional pre-loaded SMPL model (for efficiency when processing multiple clips)

        Returns:
            TT4D object (may be a failed instance if data is missing or invalid)
        """
        if smpl_model is None:
            smpl_model = cls._get_smpl()

        clip_name = gameplay_dir.split("/")[-1]

        ret = load_task(gameplay_dir, thin=True)
        (
            camera_matrix,
            rvecs,
            tvecs,
            fps,
            H_ground,
            metadata,
            _keypts_3d,
            keypts_2d,
            smpl,
            ball_positions_2d_only_tracknet,
            _frames,
            _gameplay_slice,
        ) = (
            ret["camera_matrix"],
            ret["rvecs"],
            ret["tvecs"],
            ret["fps"],
            ret["H_ground"],
            ret["metadata"],
            ret["keypts_3d"],
            ret["keypts_2d"],
            ret["smpl"],
            ret["ball_positions_2d_only_tracknet"],
            ret["frames"],
            ret["only_tracknet_second_clipped_slice"],
        )

        four_player_smpl, four_player_keypts_2d = get_four_players(metadata, smpl, keypts_2d)
        if four_player_smpl is None:
            return cls.failed("No SMPL mesh found for four players.")

        if any(len(s) == 0 for s in four_player_smpl):
            return cls.failed("No SMPL mesh found (len 0)")

        # Check for out of sync data
        if max([abs(len(four_player_smpl[i]) - ball_positions_2d_only_tracknet.shape[0]) for i in range(4)]) > 5:
            return cls.failed("Data out of sync (likely due to SMPL). Length of mismatch ...")

        # Trim all relevant lists / arrays down to the minimum available length
        # so downstream code can assume consistent per-frame indexing.
        L = min(
            min(len(s) for s in four_player_smpl),
            min(len(k) for k in four_player_keypts_2d),
            ball_positions_2d_only_tracknet.shape[0],
        )
        four_player_smpl = [s[:L] for s in four_player_smpl]
        four_player_keypts_2d = [k[:L] for k in four_player_keypts_2d]
        ball_positions_2d_only_tracknet = ball_positions_2d_only_tracknet[:L]

        # SMPL to BVH conversion per player
        four_player_bvh = []
        for player_smpl in four_player_smpl:
            go_mats, bp_mats, trans, betas = smpl_dicts_to_rotmats(player_smpl)
            pos24, _ = smpl_forward_pos(smpl_model, go_mats, bp_mats, trans, betas)
            player_bvh, _ = select_target21_from_smpl24(pos24)
            four_player_bvh.append(player_bvh)

        # Create TT4D object
        tt4d = cls(
            ball_3d=None,
            spin_3d=None,
            skeletons=four_player_bvh,
            skeletons_2d=four_player_keypts_2d,
            fps=fps,
            camera_matrix=camera_matrix,
            rvecs=rvecs,
            tvecs=tvecs,
            H_ground=H_ground,
            clip_name=clip_name,
            label="doubles",
            ball_2d=ball_positions_2d_only_tracknet,
            humans=four_player_smpl,
            ball_suffix=ball_suffix,
        )

        # Video thinning (see load_task): every per-frame feature above has the duplicated video
        # frames removed and fps is the resulting rate. A long run of duplicates means the video froze.
        if ret["frozen_run"] >= 3:
            return cls.failed(f"Video froze inside the rally ({ret['frozen_run']} consecutive duplicated frames)")
        tt4d.dropped_frames = ret["dropped_frames"]
        tt4d.video_fps = ret["video_fps"]

        tt4d.gameplay_dir = gameplay_dir
        return tt4d

    @classmethod
    def _from_gameplay_dir_doubles(cls, gameplay_dir, smpl_model=None, ball_suffix="nointerp"):
        """
        Factory method to create a TT4D object from a doubles gameplay directory.

        Args:
            gameplay_dir: Path to the gameplay directory
            smpl_model: Optional pre-loaded SMPL model (for efficiency when processing multiple clips)

        Returns:
            TT4D object (may be a failed instance if data is missing or invalid)
        """
        # Check for existing 4DHumans-based TT4D object
        clip_name = gameplay_dir.split("/")[-1]
        pkl_processed_path = os.path.join(gameplay_dir, "nima_recons", f"{clip_name}_tt4d_processed.pkl")
        if not os.path.exists(pkl_processed_path):
            return cls.failed(f"No existing TT4D result found at {pkl_processed_path}")

        if smpl_model is None:
            smpl_model = cls._get_smpl()

        # Base object
        tt4d = TT4D.load(pkl_processed_path)

        tt4d.gameplay_dir = gameplay_dir
        tt4d.foot_contact = np.zeros((tt4d.skeletons[0].shape[0], 2))

        return tt4d

    # Constructors
    @classmethod
    def from_gameplay_dir(
        cls,
        gameplay_dir,
        gameplay_file_name="",
        smpl_model=None,
        pose_source="4dhumans",
        ball_suffix="nointerp",
        needs_spark_result=False,
        full_clip=False,
    ):
        """
        Factory method to create a TT4D object from a gameplay directory.
        Automatically detects whether the gameplay is singles or doubles.

        Args:
            gameplay_dir: Path to the gameplay directory
            smpl_model: Optional pre-loaded SMPL model (for efficiency when processing multiple clips)
            pose_source: Source for human pose estimation. Options:
                - "4dhumans": Use 4DHumans output (default)
                - "genmo": Use GENMO output (only supports singles, copies pose for second player)

        Returns:
            TT4D object (may be a failed instance if data is missing or invalid)
        """
        if pose_source == "inbetweening":
            return cls._from_gameplay_dir_singles_genmo_inbetweened(
                gameplay_dir,
                gameplay_file_name,
                smpl_model,
                ball_suffix=ball_suffix,
                needs_spark_result=needs_spark_result,
            )
        elif is_singles(gameplay_dir):
            if pose_source == "genmo" and full_clip:
                return cls._from_gameplay_dir_singles_genmo_full_clip(gameplay_dir, smpl_model, ball_suffix=ball_suffix)
            if pose_source == "genmo":
                return cls._from_gameplay_dir_singles_genmo(gameplay_dir, smpl_model, ball_suffix=ball_suffix)
            else:
                return cls._from_gameplay_dir_singles(gameplay_dir, smpl_model, ball_suffix=ball_suffix)
        elif is_doubles(gameplay_dir):
            if pose_source == "genmo":
                return cls.failed("GENMO pose source not supported for doubles gameplay")
                pass
            return cls._from_gameplay_dir_doubles(gameplay_dir, smpl_model, ball_suffix=ball_suffix)
        else:
            raise cls.failed(f"Invalid gameplay directory: {gameplay_dir}")

    # Constructors
    @classmethod
    def from_gameplay_dir_partial(
        cls, gameplay_dir, gameplay_file_name="", smpl_model=None, pose_source="4dhumans", ball_suffix="nointerp"
    ):
        """
        Factory method to create a TT4D object from a gameplay directory.
        Automatically detects whether the gameplay is singles or doubles.

        Args:
            gameplay_dir: Path to the gameplay directory
            smpl_model: Optional pre-loaded SMPL model (for efficiency when processing multiple clips)
            pose_source: Source for human pose estimation. Options:
                - "4dhumans": Use 4DHumans output (default)
                - "genmo": Use GENMO output (only supports singles, copies pose for second player)

        Returns:
            TT4D object (may be a failed instance if data is missing or invalid)
        """
        # Strict agreement check: only proceed when the sb-OCR-based label and
        # the humans-mesh-based label are both decisively present and identical.
        # Anything else (humans_failed, missing humans flag, mismatch) is
        # rejected so partial TT4Ds are only built on clips with high-confidence
        # gameplay-type labels. See game_type_distinguish_humans.py.
        sb_singles = is_singles(gameplay_dir)
        sb_doubles = is_doubles(gameplay_dir)
        humans_singles = is_singles_humans(gameplay_dir)
        humans_doubles = is_doubles_humans(gameplay_dir)

        if sb_singles and humans_singles:
            if pose_source == "genmo":
                return cls._from_gameplay_dir_singles_genmo(gameplay_dir, smpl_model, ball_suffix=ball_suffix)
            else:
                return cls._from_gameplay_dir_singles_partial(gameplay_dir, smpl_model, ball_suffix=ball_suffix)
        elif sb_doubles and humans_doubles:
            if pose_source == "genmo":
                return cls.failed("GENMO pose source not supported for doubles gameplay")
            return cls._from_gameplay_dir_doubles_partial(gameplay_dir, smpl_model, ball_suffix=ball_suffix)

        sb_label = "singles" if sb_singles else ("doubles" if sb_doubles else "missing")
        humans_label = "singles" if humans_singles else ("doubles" if humans_doubles else "missing_or_failed")
        if sb_label == "missing":
            raise ValueError(f"Invalid gameplay directory (no sb singles/doubles flag): {gameplay_dir}")
        return cls.failed(f"sb label '{sb_label}' does not agree with humans label '{humans_label}'")

    def to_vae_vector(self):
        # loading
        ret = self.skeletons
        if ret is None:
            return ret
        left_seq, right_seq = ret
        assert left_seq.shape[0] == right_seq.shape[0]
        assert len(self.initial_ball_states) == 2, Exception(
            f"len(self.initial_ball_states) => {len(self.initial_ball_states)}"
        )

        # determine which skeleton and whether we need to perform a rotation
        # we want things to be on the negative x axis. this is where the motion recovery is most stable
        # left, right → negative, positive

        x_0 = self.initial_ball_states[0][0]
        if x_0[0] < 0:  # starts from negative and goes to positive
            # needs rotation
            skeleton = right_seq
            transform = R.from_euler('z', 180, degrees=True).as_matrix()
        else:
            skeleton = left_seq
            transform = R.from_euler('z', 0, degrees=True).as_matrix()

        skeleton = geometric_adjustments_smpl_to_motive(skeleton)
        skeleton = define_skeleton_relative_to_root(skeleton)
        skeleton = skeleton @ transform

        pose_and_lin_vels = np.concatenate(self.initial_ball_states, axis=0)[[0, 1, 3, 4]]
        pose_and_lin_vels = pose_and_lin_vels @ transform

        x, c = skeleton.flatten(), pose_and_lin_vels.flatten()
        if not np.isfinite(x).all() or not np.isfinite(c).all():
            print("Not all finite.")
            return None

        return x, c

    def to_full_vector(self):
        from game_dynamics.src.game_dynamics.constants import BALL_SUBS, FEATURE_DIM, POSEA_SLICE  # lives in src/dev

        # loading
        ret = self.skeletons
        if ret is None:
            return ret
        left_seq, right_seq = ret
        ball_pos = self.ball_3d
        T = left_seq.shape[0]
        assert T == right_seq.shape[0] == ball_pos.shape[0]

        # skeleton handling
        left_seq = geometric_adjustments_smpl_to_motive(left_seq)
        right_seq = geometric_adjustments_smpl_to_motive(right_seq)

        left_seq = define_skeleton_relative_to_root(left_seq)
        right_seq = define_skeleton_relative_to_root(right_seq)

        # organizing
        x = np.zeros((T, FEATURE_DIM), dtype=np.float32)
        x[:, BALL_SUBS["pos"]] = ball_pos  # no information on lin_vel or ang_vel
        x[:, POSEA_SLICE] = left_seq.reshape((T, -1))
        # x[:, POSEB_SLICE] = right_seq.reshape((T, -1))

        if not np.isfinite(x).all():
            print("Not all finite.")
            return None, None

        # False --> good data, True --> invalid data
        mask = np.isclose(x, 0)
        return x, mask

    def split_full_vector(x_denorm: np.ndarray):
        from game_dynamics.src.game_dynamics.constants import BALL_SLICE, POSEA_SLICE  # lives in src/dev

        ball = x_denorm[:, BALL_SLICE]
        poseA = x_denorm[:, POSEA_SLICE]
        # poseB = x_denorm[:, POSEB_SLICE]
        return ball, poseA, poseB
