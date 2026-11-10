"""Ball trajectory cleaning, hit-point detection, ODE fitting and the inverse racket estimate."""

import numpy as np
from scipy.ndimage import label
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation as R

from rtt_utils.constants import TABLE_HEIGHT
from rtt_utils.ode_fit import TABLE_Z, filter_obs_and_time, fit_initial_state, simulate_with_bounces_fast
from rtt_utils.racket_model.solve_stroke_single_shooting import solve_stroke


def interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate NaNs along axis 0 (row-wise) for each column.
    Works for shape (N, D).
    """
    arr_out = arr.copy()
    n = arr.shape[0]
    x = np.arange(n)

    for j in range(arr.shape[1]):  # loop over columns
        col = arr[:, j]
        mask = np.isfinite(col)
        if mask.any():  # only interpolate if there are some finite values
            arr_out[:, j] = np.interp(x, x[mask], col[mask])
    return arr_out


def clean_ball_segment(ball_xyz, FPS):
    """
    Fit the flight ODE to one segment of 3D ball positions and resimulate it on every frame.

    Returns (initial state [p0, v0, w0], interpolated (T, 3) trajectory, squared fit error); the state and
    the error are None and the trajectory all NaN when too few points remain or the fit fails.
    """
    ball_xyz = ball_xyz.copy()
    dev = np.linalg.norm(np.diff(ball_xyz, axis=0), axis=1)  # length T-1
    speed = dev * FPS
    mask = np.r_[False, speed > 100]  # arrival frame mask, shape T

    # Drop a "bomb" around the bad point ... destroys everything in a local neighborhood
    # Expand/dilate mask by ±k frames
    k = 4  # number of neighbors to also NaN-out
    for shift in range(-k, k + 1):
        if shift == 0:
            continue
        mask |= np.roll(mask, shift)
    ball_xyz[mask] = np.nan

    # TT3D cleaning algorithm (Gossard et al.)
    # INPUT (N, 3) array for ball positions
    # occluded points are represented as (0, 0, 0)

    # replace nans with (0, 0, 0)
    ball_xyz = np.where(np.isnan(ball_xyz), 0, ball_xyz)

    # 1) Filter P and t with SAME mask (preserves original timing)
    t_original = np.arange(len(ball_xyz)) / FPS
    t, P_obs = filter_obs_and_time(ball_xyz, FPS)

    if P_obs.shape[0] < 2:
        print("[clean_ball_segment] Not enough points.")
        return None, np.full(shape=ball_xyz.shape, fill_value=np.nan), None

    # 2) Initialize from the first two kept samples (their original dt)
    p0_init = P_obs[0]
    dt01 = max(t[1] - t[0], 1e-3)
    v0_init = (P_obs[1] - P_obs[0]) / dt01
    w0_init = np.zeros(3, dtype=np.float64)

    try:
        # 3) Fit
        res = fit_initial_state(
            t,
            P_obs,
            p0_init=p0_init,
            v0_init=v0_init,
            w0_init=w0_init,
            table_z=TABLE_Z,
            max_bounces=2,
            substep=0.002,
            robust_huber=True,
        )
    except Exception:
        print("[clean_ball_segment] Fit failed")
        return None, np.full(shape=ball_xyz.shape, fill_value=np.nan), None

    rmse = float(np.sqrt(np.mean((res.traj_pred - P_obs) ** 2)))
    if rmse > 100:
        print(f"[clean_ball_segment] rmse too high {rmse}")
        return None, np.full(shape=ball_xyz.shape, fill_value=np.nan), None

    initial_ball_state = [res.p0, res.v0, res.w0]
    interpolated = simulate_with_bounces_fast(
        *initial_ball_state,
        t_original,
        table_z=TABLE_Z,
    )

    assert ball_xyz.shape == interpolated.shape

    error = (res.traj_pred - P_obs) ** 2
    return initial_ball_state, interpolated, error


def inverse_racket_estimation(ball_xyz, seg_start, seg_end, FPS):
    """
    Solve for the racket pose and velocity at the hit that starts segment [seg_start, seg_end).

    Returns (hit position, racket velocity, racket quaternion xyzw, meta) in the pipeline's world frame, or
    four Nones when the segment has no single bounce. The solver works in the TT3D table frame (x along the
    width, surface at z=0) and only for a ball travelling in +y, so inputs are converted and flipped back.
    """
    print(f"inverse_racket_estimation for {seg_start, seg_end}")
    ball_xyz_segment = ball_xyz[seg_start:seg_end]
    z_segment = ball_xyz[seg_start:seg_end, 2]

    pre_impact_ball_pos = ball_xyz[seg_start - int(FPS * 0.05) : seg_start]
    avg_preimpact_ball_lin_vel = (
        np.mean(np.diff(pre_impact_ball_pos, axis=0), axis=0) * FPS
    )  # meters per sample --> meters per second

    possible_bounce_indices = z_segment <= 0.84
    _, n_labels = label(possible_bounce_indices)
    if n_labels != 1:
        print("not n_labels == 1")
        return None, None, None, None
    bounce_index = np.argmin(z_segment)
    bounce_position = ball_xyz_segment[bounce_index]
    tof = bounce_index / FPS

    # transform coordinates to the TT3D table frame
    R_tt3d_table = np.array(
        [
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]
    )
    t_tt3d_table = np.array([0, 0, -TABLE_HEIGHT])
    hit_pos = R_tt3d_table @ ball_xyz[seg_start] + t_tt3d_table
    vb_in = R_tt3d_table @ avg_preimpact_ball_lin_vel
    target_pos = R_tt3d_table @ bounce_position + t_tt3d_table
    wb_in = np.zeros(3)
    target_spin = np.zeros(3)

    flip = vb_in[1] > 0  # if moving from pos to neg, switch such that it moves from neg to pos
    if flip:
        print("flip")
        # The TT3D solver only works in one direction. Need to adjust for this
        # was moving right before the contact. after the contact, it's moving left.
        # adjust the input such that after the contact, it's moving right.
        hit_pos[0], hit_pos[1] = -hit_pos[0], -hit_pos[1]  # rotation about the z axis
        vb_in[0], vb_in[1] = -vb_in[0], -vb_in[1]
        target_pos[0], target_pos[1] = -target_pos[0], -target_pos[1]

    result = solve_stroke(
        hit_pos=hit_pos,
        vb_in=vb_in,
        wb_in=wb_in,
        target_pos=target_pos,
        target_spin=target_spin,
        tf=tof,
        use_T_flight=True,
    )
    racket_vel, racket_quat = result["v_racket"], result["q_racket"]  # these should be in the table frame

    if flip:
        print("flip back")
        hit_pos[0], hit_pos[1] = -hit_pos[0], -hit_pos[1]
        vb_in[0], vb_in[1] = -vb_in[0], -vb_in[1]
        target_pos[0], target_pos[1] = -target_pos[0], -target_pos[1]
        racket_vel[0], racket_vel[1] = -racket_vel[0], -racket_vel[1]
        racket_quat = (R.from_euler('z', 180, degrees=True) * R.from_quat(racket_quat)).as_quat()

    # convert back to the pipeline's world frame by using the inverse of R and t above
    hit_pos = R_tt3d_table.T @ hit_pos - R_tt3d_table.T @ t_tt3d_table
    racket_vel = R_tt3d_table.T @ racket_vel
    racket_quat = (R.from_matrix(R_tt3d_table.T) * R.from_quat(racket_quat)).as_quat()

    meta = dict()
    meta["hit_pos"] = ball_xyz[seg_start]
    meta["vel_in"] = avg_preimpact_ball_lin_vel
    meta["target_pos"] = bounce_position
    meta["result"] = result

    print("####################################################################")

    return hit_pos, racket_vel, racket_quat, meta


def detect_bounces_in_segment(z, height=TABLE_HEIGHT + 0.15, distance=6):
    """Return indices of bounce points (local minima in z near the table)."""
    peaks, _ = find_peaks(-z, height=-height, distance=distance)
    return peaks


def second_clipping_refinement(recons, gameplay_dir=None):
    """
    Find the hits of a rally as the alternating extremes of the ball's x coordinate.

    Returns (first hit index, last hit index, all hit indices) into recons["ball_3d"], or None when fewer than
    two alternating extremes are found.
    """
    ball_3d_full = recons["ball_3d"]  # corresponds to ball_tracker_only_tracknet

    # Keep this mask + index map so we can go back to original indices later
    not_vis_mask = np.all(ball_3d_full == 0, axis=1)
    assert not_vis_mask.sum() == 0  # everything is visible when we have an interpolation result
    keep_mask = ~not_vis_mask
    idx_map = np.flatnonzero(keep_mask)  # idx in original -> filtered

    ball_3d = ball_3d_full[keep_mask]

    if ball_3d.shape[0] == 0:
        return

    x = ball_3d[:, 0]

    ######################## PEAK REFINEMENT #####################

    rng = np.ptp(x) if np.ptp(x) > 0 else 1.0
    PEAK_HEIGHT = 0.3
    peaks, _ = find_peaks(
        x, height=PEAK_HEIGHT, prominence=0.02 * rng, distance=6
    )  # oh, this is distance from peak to peak, not distance from peak to trough ... i initially had this at 3, but I can actually bump it to double that
    troughs, _ = find_peaks(-x, height=PEAK_HEIGHT, prominence=0.02 * rng, distance=6)

    idx_extremes = np.sort(np.concatenate([peaks, troughs]))
    extreme = ball_3d[idx_extremes, 0]

    if len(extreme) == 0:
        return

    # from the back
    # remove all but the last extreme if the signs are repeated
    # e.g. (1, -1, 1, 1, 1, 1) --> (1, -1, 1) with three repeated
    signs = np.sign(extreme)[::-1]
    s_init = signs[0]
    repeated = 0
    for s in signs[1:]:
        if s == s_init:
            repeated += 1
        else:
            break
    if repeated > 0:
        idx_extremes = idx_extremes[:-repeated]
        extreme = ball_3d[idx_extremes, 0]

    if len(extreme) == 0:
        return

    # from the front
    # remove the repetitions after the first one that we identify maybe????
    # e.g. (1, 1, 1, -1, 1, -1, 1, -1) --> (1, -1, 1) with three repeated
    signs = np.sign(extreme)
    s_init = signs[0]
    repeated = 0
    for s in signs[1:]:
        if s == s_init:
            repeated += 1
        else:
            break
    idx_extremes = idx_extremes[repeated:]
    extreme = ball_3d[idx_extremes, 0]

    # stop when the signs repeat
    repeated_signs = np.sign(extreme[:-1]) == np.sign(extreme[1:])
    true_indices = np.where(repeated_signs)[0]
    idx_first_true = true_indices[0] if true_indices.size > 0 else None
    if idx_first_true:
        idx_extremes = idx_extremes[: idx_first_true + 1]
        extreme = ball_3d[idx_extremes, 0]

    extremes_beyond_threshold = extreme.shape[0]
    if extremes_beyond_threshold <= 1:
        return

    # Map filtered indices back to original frame indices
    orig_idx_extremes = idx_map[idx_extremes]
    orig_start, orig_end = orig_idx_extremes[0], orig_idx_extremes[-1]

    return orig_start, orig_end, orig_idx_extremes


def classify_spin(spin, threshold=5):
    """
    Classify the spin into top, back, side_left, side_right, no_spin.
    """
    if np.abs(spin[1]) > threshold:
        if spin[1] > 0:
            return "top"
        elif spin[1] < 0:
            return "back"
    elif np.abs(spin[2]) > threshold:
        if spin[2] > 0:
            return "side_left"
        elif spin[2] < 0:
            return "side_right"
    else:
        return "no_spin"
