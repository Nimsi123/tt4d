"""Find the periodically duplicated frames of a video (e.g. 25 fps footage stored at 30 fps) from frame-to-frame SSIM."""

import math

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

SSIM_THRESHOLD_DEFAULT = 0.995  # Threshold to flag "same" frames (1.0 = identical) only valid for 30fps
MIN_ALLOWED_FREQ = 3  # Minimum frequency allowed


def get_video_props(video_path):
    """Return (fps, frame count, (width, height)) from the container metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n_frames, (w, h)


def compute_ssim_scores(video_path):
    """
    Compute SSIM scores between consecutive frames.
    Returns a numpy array of SSIM scores (len = n_frames - 1).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Failed to open video: {video_path}")

    prev_gray = None
    scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            score = ssim(prev_gray, gray)
            scores.append(score)
        prev_gray = gray

    cap.release()
    scores = np.array(scores, dtype=float)
    return scores


def find_duplicate_indices(scores, ssim_threshold=SSIM_THRESHOLD_DEFAULT):
    """
    Given SSIM scores, return indices of frames that are duplicates of the previous one.
    (Index i corresponds to frame i being similar to frame i-1.)
    """
    return np.where(scores >= ssim_threshold)[0] + 1  # +1 because score[i] = ssim(frame[i], frame[i-1])


def estimate_start_and_freq(dup_idx):
    """
    Given duplicate indices dup_idx ~ start + k*freq, estimate (start, freq).
    Returns ints (start, freq).
    """
    if len(dup_idx) < 3:
        raise ValueError("Not enough repeating indices to estimate start/freq (need >= 3).")
    diffs = np.diff(dup_idx).astype(int)
    diffs = diffs[diffs > 1]  # ignore immediate repeats
    if len(diffs) == 0:
        return 0, 1
    # modal spacing
    freq_est = int(np.argmax(np.bincount(diffs)))
    # most common remainder -> start
    remainders = dup_idx % freq_est
    start_est = int(np.bincount(remainders).argmax())
    return start_est, freq_est


def indices_to_drop(n_frames, start, freq):
    """
    Produce a set of integer frame indices to drop:
      drop i if i == start + m*freq for some integer m
    """
    drop = set()
    if freq <= 1 or not np.isfinite(freq):
        return drop
    m_min = math.floor((0 - start) / freq) - 1
    m_max = math.ceil((n_frames - 1 - start) / freq) + 1
    for m in range(m_min, m_max + 1):
        target = start + m * freq
        if 0 <= target < n_frames and float(target).is_integer():
            drop.add(int(target))
    return drop


def duplicate_frame_indices(
    video_file: str,
    ssim_threshold: float = SSIM_THRESHOLD_DEFAULT,
    min_allowed_freq: int = MIN_ALLOWED_FREQ,
) -> list[int]:
    """
    Return a sorted list of frame indices that are duplicates and can be dropped.

    Near-identical consecutive frames (SSIM above the threshold) are duplicates. A regular cadence of them
    (at most three distinct gaps) is treated as periodic and extrapolated over the whole video; an irregular
    but sparse pattern drops the detected frames only; a dense irregular pattern drops nothing.
    """
    fps, n_frames, _ = get_video_props(video_file)

    scores = compute_ssim_scores(video_file)
    # Slightly stricter threshold for high-FPS videos
    ssim_thresh = 0.998 if fps > 45 else ssim_threshold
    dup_idx = find_duplicate_indices(scores, ssim_threshold=ssim_thresh)

    # Not enough evidence → no removal
    if len(dup_idx) < 4:
        return []

    # Examine gaps between duplicate indices (ignore immediate repeats)
    diff_dup_idx = np.diff(dup_idx)
    diff_dup_idx = diff_dup_idx[diff_dup_idx > 1]

    if len(diff_dup_idx) == 0:
        # Only consecutive duplicates → treat as non-periodic
        return []

    uniq = np.unique(diff_dup_idx)

    if len(uniq) > 5:
        # Highly irregular → skip removal
        return []

    if len(uniq) > 3:
        # Irregular but not crazy → remove detected aperiodic duplicates directly
        return sorted(int(i) for i in dup_idx.tolist())

    # Looks periodic → estimate start/freq
    start_est, freq_est = estimate_start_and_freq(dup_idx)

    # Validate frequency
    if (not np.isfinite(freq_est)) or (freq_est < min_allowed_freq) or (freq_est <= 1):
        return []

    # Build periodic drop set
    drop = indices_to_drop(n_frames, int(round(start_est)), int(round(freq_est)))
    return sorted(drop)
