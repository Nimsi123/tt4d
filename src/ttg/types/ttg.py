# Standard library imports
import os
import pickle

# Third-party imports
import numpy as np

from rtt_utils.ball import clean_ball_segment, detect_bounces_in_segment, interpolate_nans, inverse_racket_estimation

# Local/application imports
from rtt_utils.calibration import compute_table_diagonal_pixels
from rtt_utils.constants import TABLE_LENGTH, TABLE_WIDTH


class TTG:
    def __init__(self, ball_3d, skeletons, fps):
        self.ball_3d = ball_3d

        for s in skeletons:
            for j in range(s.shape[1]):
                s[:, j, :] = interpolate_nans(s[:, j, :])
        self._skeletons = skeletons
        for s in self._skeletons:
            if np.isnan(s).sum() > 0:
                return TTG.failed("nans in skeleton")

        self._processed_skeletons = None
        self.fps = fps
        self.estimated_paddles = None

        self.reprojection_errors_neural = None

        self.invalid = False
        self.failure_reason = None

    @classmethod
    def failed(cls, failure_reason):
        """Create a TT4D object representing a failed reconstruction."""
        instance = cls.__new__(cls)
        instance.invalid = True
        instance.failure_reason = failure_reason
        # Set all other attributes to None
        instance.ball_3d = None
        instance._skeletons = None
        instance._processed_skeletons = None
        instance.fps = None
        instance.estimated_paddles = None
        instance.reprojection_errors_neural = None

        return instance

    @property
    def skeletons(self):
        #
        """
        _skeletons is the result of select_target21_from_smpl24


        MOTIVE21_NAMES = [
            "Hips",
            "Spine","Spine1","Neck","Head",
            "LeftShoulder","LeftArm","LeftForeArm","LeftHand",
            "RightShoulder","RightArm","RightForeArm","RightHand",
            "LeftUpLeg","LeftLeg","LeftFoot","LeftToeBase",
            "RightUpLeg","RightLeg","RightFoot","RightToeBase",
        ]
        """

        # (N, 21, 3)

        # don't include this. leads to tricky bugs (even if we slice, the skeleton won't be sliced)
        # if self._processed_skeletons is not None:
        #     return self._processed_skeletons

        if hasattr(self, "pose_source") and self.pose_source == "genmo":
            # 1/0
            return self._skeletons

        if len(self._skeletons) > 2:
            return self._skeletons[:2]

        left_seq, right_seq = self._skeletons

        T = left_seq.shape[0]

        left_seq = left_seq.reshape(T, -1, 3).astype(np.float32)
        right_seq = right_seq.reshape(T, -1, 3).astype(np.float32)

        if (left_seq[:, 0, 0] < 0).sum() > 0 and (left_seq[:, 0, 0] > 0).sum() > 0:
            print("Inconsistent assignment of left and right poses.")
            self.invalid = True
            self.failure_reason = "Inconsistent assignment of left and right poses."
            return
        # I FEEL LIKE I SHOULDN"T NEED THIS!!!! THERE IS A BUG SOMEWHERE!!! I DIDN'T NEED THIS BEFORE!!!
        if (left_seq[:, 0, 0] > 0).all() and (right_seq[:, 0, 0] < 0).all():
            self.invalid = True
            self.failure_reason = "Improper assignment of left and right poses"
            raise Exception("Improper assignment of left and right poses")
        assert (left_seq[:, 0, 0] < 0).all()
        if T > 10:
            # filter ... homography makes things noisy. we saw this with normal vector estimation of racket
            w = np.array([1, 1, 1, 1], dtype=np.float32) / 4.0  # box filter
            for ax in range(3):
                # Pad by repeating the boundary values
                left_pad = np.pad(left_seq[:, 0, ax], ((len(w) - 1) // 2, len(w) // 2), mode='edge')
                left_seq[:, 0, ax] = np.convolve(left_pad, w, mode="valid")
                right_pad = np.pad(right_seq[:, 0, ax], ((len(w) - 1) // 2, len(w) // 2), mode='edge')
                right_seq[:, 0, ax] = np.convolve(right_pad, w, mode="valid")

        self._processed_skeletons = [left_seq, right_seq]

        return self._processed_skeletons

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save_npz(self, path):
        assert "MotiveGameplay" in str(self.__class__) or self.label == "singles", "Only singles are supported for now"

        recon = {
            "ball_3d": self.ball_3d,
            "spin_3d": self.spin_3d,
            "negative_player_bvh": self.skeletons[0],
            "positive_player_bvh": self.skeletons[1],
            "fps": self.fps,
            "hit_indices": self.hit_indices,
        }
        if int(self.fps) == 360:
            from scipy.signal import decimate

            factor = int(self.fps) // 30  # 12

            for key in ("ball_3d", "spin_3d", "negative_player_bvh", "positive_player_bvh"):
                arr = recon[key]
                orig_shape = arr.shape
                flat = arr.reshape(arr.shape[0], -1)  # (T, D)
                flat = decimate(flat, factor, axis=0)
                recon[key] = flat.reshape(-1, *orig_shape[1:])

            recon["hit_indices"] = np.round(np.asarray(recon["hit_indices"], dtype=np.float64) / factor).astype(
                np.int64
            )
            recon["fps"] = 30

        np.savez_compressed(path, **recon)

    def set_hit_indices(self, hit_indices):
        """
        Set pre-computed hit indices directly, skipping automatic hit point identification.

        Args:
            hit_indices: List/array of frame indices where hits occur.
        """
        self.hit_indices = list(hit_indices)
        self.segments = list(zip(hit_indices[:-1], hit_indices[1:]))
        self.exchanges = [(s1[0], s2[1]) for s1, s2 in zip(self.segments[:-1], self.segments[1:])]

    def annotate_ode_fit(self):
        self.initial_ball_states = []
        self.ode_fit_errors = []
        self.interpolated_ball_3d = np.full_like(self.ball_3d, np.nan)
        for start, end in self.segments:
            initial_ball_state, ball_3d, fit_error = clean_ball_segment(self.ball_3d[start:end], self.fps)
            self.initial_ball_states.append(initial_ball_state)
            self.interpolated_ball_3d[start:end] = ball_3d
            self.ode_fit_errors.append(fit_error)

    def determine_validity(self, MAX_RMSE_THRESHOLD, MAX_NEURAL_REPROJECTION_THRESHOLD, stats=False):
        if self.invalid:
            if not self.failure_reason:
                self.failure_reason = "Failure by initialization."
            return False

        # --- Video checks ---
        # 25-30 covers the thinned rate (duplicated frames removed, see load_task)
        if not stats and not (25 <= self.fps <= 30 or self.fps == 360):
            self.failure_reason = f"FPS of {self.fps} is not supported."
            return False

        # --- Ball checks ---
        if len(self.ode_fit_errors) == 0:
            self.failure_reason = "ODE fit too high"
            self.invalid = True
            return False
        ode_fit_rmse = []
        for e in self.ode_fit_errors:
            if e is None:
                self.failure_reason = "ODE fit too high"
                self.invalid = True
                return False
            else:
                rmse = float(np.sqrt(np.mean(e)))
                ode_fit_rmse.append(rmse)
        if np.max(ode_fit_rmse) > MAX_RMSE_THRESHOLD:
            self.failure_reason = "ODE fit too high"
            self.invalid = True
            return False

        if self.reprojection_errors_neural is not None:
            reprojection_errors = []
            for e in self.reprojection_errors_neural:
                reprojection_errors.append(np.mean(np.sum(e, axis=1)))
            table_diagonal_pixels = compute_table_diagonal_pixels(self.rvecs, self.tvecs, self.camera_matrix)
            normalized_max_reprojection_error = max(reprojection_errors) / table_diagonal_pixels

            if (
                normalized_max_reprojection_error > MAX_NEURAL_REPROJECTION_THRESHOLD
                # or max(errors_interpolation) > MAX_INTERPOLATION_REPROJECTION_THRESHOLD
            ):
                self.failure_reason = "Reprojection error too high"
                self.invalid = True
                return False

        # --- Skeleton checks ---
        if np.isnan(self._skeletons[0]).any() or np.isnan(self._skeletons[1]).any():
            self.failure_reason = "At least one skeleton is missing"
            self.invalid = True
            return False

        skels = self.skeletons  # has embedded checks
        if skels is None:
            self.invalid = True
            self.failure_reason = "No skeletons."
            return False

        for sk in skels:
            maximum_absolute_deviation_m_per_sample = np.abs(sk[:-1] - sk[1:]).max()
            maximum_absolute_deviation_m_per_sec = maximum_absolute_deviation_m_per_sample * self.fps
            if maximum_absolute_deviation_m_per_sec > 1e10:
                self.failure_reason = "Skeleton changed too fast"
                self.invalid = True
                return False

        # additional checks
        if len(self.segments) < 2:
            self.invalid = True
            self.failure_reason = "Not enough segments."
            return False

        # SKELETON x position check
        # I think there are some issues that still need to be resolved ...
        # The position is not as reliable as I'd like
        # Oh, maybe it's because the debug_clipping.ipynb is using the 4DHumans result
        if np.any(np.abs(self.skeletons[0][:, 0, 0]) < 0.7) or np.any(np.abs(self.skeletons[1][:, 0, 0]) < 0.7):
            self.invalid = True
            self.failure_reason = "Invalid skeleton x position (too close to table)."
            return False

        # SKELETON y position check
        # honestly, this should be done after we place the humans
        # y_pos_threshold = 1.8
        y_pos_threshold = 2.5
        first_seg = self.segments[0]
        if np.any(np.abs(self.skeletons[0][first_seg[0] : first_seg[1], 0, 1]) > y_pos_threshold) or np.any(
            np.abs(self.skeletons[1][first_seg[0] : first_seg[1], 0, 1]) > y_pos_threshold
        ):
            self.invalid = True
            self.failure_reason = "Invalid skeleton y position (too far) in the first segment."
            return False
        last_seg = self.segments[-1]
        if np.any(np.abs(self.skeletons[0][last_seg[0] : last_seg[1], 0, 1]) > y_pos_threshold) or np.any(
            np.abs(self.skeletons[1][last_seg[0] : last_seg[1], 0, 1]) > y_pos_threshold
        ):
            # cut the last segment and retry
            self.slice_gameplay(end_idx=last_seg[0] + 1)

            if np.any(np.abs(self.skeletons[0][last_seg[0] : last_seg[1], 0, 1]) > y_pos_threshold) or np.any(
                np.abs(self.skeletons[1][last_seg[0] : last_seg[1], 0, 1]) > y_pos_threshold
            ):
                self.invalid = True
                self.failure_reason = "Invalid skeleton y position (too far) in the last segment."
                return False

        # SKELETON z position check
        if np.any(self.skeletons[0][:, 0, 2] < 0.3) or np.any(self.skeletons[0][:, 0, 2] > 1.5):
            self.invalid = True
            self.failure_reason = "Invalid skeleton z position for skeleton 0."
            return False

        if np.any(self.skeletons[1][:, 0, 2] < 0.3) or np.any(self.skeletons[1][:, 0, 2] > 1.5):
            self.invalid = True
            self.failure_reason = "Invalid skeleton z position for skeleton 1."
            return False

        # BALL position check
        # ball can't be under the table
        if np.any(
            (np.abs(self.ball_3d[:, 0]) < TABLE_LENGTH / 2 - 0.05)
            & (np.abs(self.ball_3d[:, 1]) < TABLE_WIDTH / 2 - 0.05)
            & (self.ball_3d[:, 2] < 0.65)
        ):
            self.invalid = True
            self.failure_reason = "Invalid ball position (under the table)."
            return False

        # LAST SEGMENT CHECKS

        # trim the last segment
        last_seg = self.segments[-1]
        balls_under_table = np.count_nonzero(self.ball_3d[last_seg[0] : last_seg[1], 2] < 0.6)
        if balls_under_table > 5:
            self.slice_gameplay(end_idx=last_seg[0] + 1)

        # exclude the segments (and everything after) when there's a double bounce
        # i feel like this would happen towards the end of the point (at the last segment)
        for seg in self.segments[1:]:
            z_seg = self.ball_3d[seg[0] : seg[1], 2]
            bounce_idxs = detect_bounces_in_segment(z_seg)
            if len(bounce_idxs) > 1:
                self.slice_gameplay(end_idx=seg[0] + 1)
                break

        # minimum segment length
        if len(self.segments) < 2:
            self.invalid = True
            self.failure_reason = "Not enough segments (after further trimming)."
            return False

        ball_vis_thresh = 0.5
        if len(self.segments) < 4 and (self.ball_2d[:, 1].mean() < ball_vis_thresh):
            self.invalid = True
            self.failure_reason = f"Ball visibility below threshold: {ball_vis_thresh}."
            return False

        # Skeleton x check
        if np.all(self.skeletons[0][:, 0, 0] > 0) and np.all(self.skeletons[1][:, 0, 0] < 0):
            self.invalid = True
            self.failure_reason = (
                "Invalid skeleton x position (all wrong sides). These are likely doubles gameplay labeled as singles."
            )
            return False
        if np.any(self.skeletons[0][:, 0, 0] > 0) or np.any(self.skeletons[1][:, 0, 0] < 0):
            self.invalid = True
            self.failure_reason = "Invalid skeleton x position (some wrong sides)."
            return False

        return True

    def add_racket_annotations(self):
        # only depends on the segments and ball_3d

        # if ...:
        #     print("Racket annotations have already been added!")
        #     return

        N = self.time_s.shape[0]
        est_paddle_pos_xyz_m = np.full((N, 3), np.nan, float)
        est_paddle_lin_vel_xyz_m = np.full((N, 3), np.nan, float)
        est_paddle_quat_xyzw = np.full((N, 4), np.nan, float)
        est_paddle_score = np.full((N,), np.nan, float)
        est_paddle_meta = [None] * N

        for segment in self.segments[1:]:
            racket_point, racket_vel, racket_quat, meta = inverse_racket_estimation(
                self.interpolated_ball_3d, segment[0], segment[1], self.fps
            )

            if all([r is None for r in [racket_point, racket_vel, racket_quat]]):
                print("continue ...")
                continue

            # abs_start, abs_end = pt[0] + segment[0], pt[0] + segment[1]
            abs_start, abs_end = segment[0], segment[1]
            # constant for the duration of the segment
            est_paddle_pos_xyz_m[abs_start:abs_end] = racket_point
            est_paddle_lin_vel_xyz_m[abs_start:abs_end] = racket_vel
            est_paddle_quat_xyzw[abs_start:abs_end] = racket_quat
            est_paddle_score[abs_start:abs_end] = 0
            for i in range(abs_start, abs_end):
                est_paddle_meta[i] = meta

        # Store the computed data
        self.est_paddle_pos_xyz_m = est_paddle_pos_xyz_m
        self.est_paddle_lin_vel_xyz_m = est_paddle_lin_vel_xyz_m
        self.est_paddle_quat_xyzw = est_paddle_quat_xyzw
        self.est_paddle_score = est_paddle_score
        self.est_paddle_meta = est_paddle_meta
