"""
Script to calibrate the camera using Nima's dataset.
Performs calibration for each video frame, filters results, checks for motion,
and writes out final calibration data and visualizations.
"""

import argparse
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

from table_calibrator import TableCalibrator
from utils import save_camcal, write_camera_info
from filter import interquartile_mean, KalmanFilterWithOutlierRejection

# 3D coordinates of table corners in the world frame (in meters)
TABLE_CORNERS_3D = np.array(
    [[-0.7625, 1.37, 0], [0.7625, 1.37, 0], [0.7625, -1.37, 0], [-0.7625, -1.37, 0]]
)


def get_frame(video_path, frame_idx=50):
    """Extracts a single frame from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx >= total_frames:
        raise IndexError(f"Frame index {frame_idx} out of range.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame at index {frame_idx}.")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def check_camera_motion(f_rvecs, f_tvecs, f_fs, trans_ratio_thresh=0.5, rot_thresh=0.2):
    """Determines if the camera moved based on change in orientation and tz/f ratio."""
    tz_over_f = f_tvecs[:, 2] / f_fs
    delta_r = np.linalg.norm(np.diff(f_rvecs, axis=0), axis=1)
    delta_tz_over_f = np.abs(np.diff(tz_over_f))

    max_rot = np.max(delta_r)
    max_tz_ratio = np.max(delta_tz_over_f)
    is_moving = (max_rot > rot_thresh) or (max_tz_ratio > trans_ratio_thresh)

    return is_moving, {"max_rotation": max_rot, "max_tz_over_f": max_tz_ratio}


def plot_calibration(rvecs, tvecs, fs, fig_path):
    """Plots raw and filtered calibration data for visual inspection."""
    idxs = np.arange(len(rvecs))
    valid_mask = ~(
        (np.all(rvecs == 0, axis=1)) | (np.all(tvecs == 0, axis=1)) | (fs == 0)
    )
    valid_idxs = np.where(valid_mask)[0]

    rvecs_valid, tvecs_valid, fs_valid = (
        rvecs[valid_mask],
        tvecs[valid_mask],
        fs[valid_mask],
    )

    # Filter data using Kalman filter with outlier rejection
    f_rvecs = KalmanFilterWithOutlierRejection(n_dim=3).fit(rvecs_valid)
    f_tvecs = KalmanFilterWithOutlierRejection(n_dim=3).fit(tvecs_valid)
    f_f = KalmanFilterWithOutlierRejection(n_dim=1).fit(fs_valid)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # Focal length
    axs[0].scatter(idxs, fs, color="purple", alpha=0.4, label="f (raw)")
    axs[0].plot(valid_idxs, f_f, color="purple", label="f (filtered)")
    axs[0].set_ylabel("f")
    axs[0].grid(True)

    # Rotation vectors
    for i, label, color in zip(range(3), ["X", "Y", "Z"], ["red", "green", "blue"]):
        axs[1].scatter(
            idxs, rvecs[:, i], label=f"Rvec_{label} (raw)", color=color, alpha=0.4
        )
        axs[1].plot(
            valid_idxs,
            f_rvecs[:, i],
            label=f"Rvec_{label} (filtered)",
            color=f"dark{color}",
        )

    axs[1].set_ylabel("Rvec [rad]")
    axs[1].grid(True)

    # Translation vectors
    for i, label, color in zip(range(3), ["X", "Y", "Z"], ["red", "green", "blue"]):
        axs[2].scatter(
            idxs, tvecs[:, i], label=f"Tvec_{label} (raw)", color=color, alpha=0.5
        )
        axs[2].plot(valid_idxs, f_tvecs[:, i], color=f"dark{color}")

    axs[2].set_xlabel("Frame")
    axs[2].set_ylabel("Tvec [m]")
    axs[2].grid(True)

    axs[2].legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=3)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close("all")

    return valid_idxs, f_rvecs, f_tvecs, f_f


def calibrate_video(input_dir, device):
    """Main calibration pipeline for all frames of a video."""
    input_dir = Path(input_dir)
    # Pipeline clips keep the video in <clip>/video/; the tt3d evaluation clips keep it at the top level.
    video_path = (list((input_dir / "video").glob("*.mp4")) or list(input_dir.glob("*.mp4")))[0]
    # video_path = list((input_dir / "video_second_clipped").glob("*.mp4"))[0]
    output_dir = input_dir / "thomas_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load a reference frame
    ref_frame = get_frame(video_path)

    # Set up video capture and calibrator
    cap = cv2.VideoCapture(str(video_path))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    calibrator = TableCalibrator(height, width, device=device)

    rvecs, tvecs, fs, ers = [], [], [], []

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rvec, tvec, f, er, _ = calibrator.process(frame, debug=True)
        rvecs.append(rvec.flatten() if rvec is not None else np.zeros(3))
        tvecs.append(tvec.flatten() if tvec is not None else np.zeros(3))
        fs.append(f if f is not None else 0)
        ers.append(er if er is not None else 0)

    cap.release()

    # Convert to arrays and plot
    rvecs = np.array(rvecs)
    tvecs = np.array(tvecs)
    fs = np.array(fs)
    ers = np.array(ers)
    fig_path = output_dir / "calib_plot.png"
    valid_idxs, f_rvecs, f_tvecs, f_f = plot_calibration(rvecs, tvecs, fs, fig_path)

    # Not enough good estimates
    if len(valid_idxs) < 40:
        (output_dir / "failed.txt").write_text("Not enough good estimates")
        return

    # Save raw calibration data
    save_camcal(output_dir / "camcal.csv", rvecs, tvecs, fs, errors=ers)

    # Motion detection
    is_moving, movement = check_camera_motion(f_rvecs, f_tvecs, f_f)
    if is_moving:
        with open(output_dir / "moves.txt", "w") as f:
            for k, v in movement.items():
                f.write(f"{k}: {v:.6f}\n")
    else:
        # Save static camera calibration result
        m_rvec, m_tvec, m_f = (
            interquartile_mean(f_rvecs),
            interquartile_mean(f_tvecs),
            interquartile_mean(f_f),
        )
        write_camera_info(output_dir / "calib.yaml", m_rvec, m_tvec, m_f, height, width)

        # Reproject table corners and axes
        K = np.array([[m_f, 0, width / 2], [0, m_f, height / 2], [0, 0, 1]])
        dist = np.zeros(5)

        projected_corners = [
            cv2.projectPoints(corner, m_rvec, m_tvec, K, dist)[0][0][0]
            for corner in TABLE_CORNERS_3D
        ]
        for pt in projected_corners:
            cv2.circle(ref_frame, tuple(map(int, pt)), 3, (255, 0, 0), -1)

        origin = np.array([[0, 0, 0.0001]])
        axes = (
            [origin + np.array([[s, 0, 0]]) for s in [0.7625]]
            + [origin + np.array([[0, s, 0]]) for s in [1.37]]
            + [origin + np.array([[0, 0, s]]) for s in [1.0]]
        )

        projected_axes = [
            cv2.projectPoints(axis, m_rvec, m_tvec, K, dist)[0][0][0] for axis in axes
        ]
        origin_pt = cv2.projectPoints(origin, m_rvec, m_tvec, K, dist)[0][0][0]
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

        for axis_end, color in zip(projected_axes, colors):
            cv2.line(
                ref_frame,
                tuple(map(int, origin_pt)),
                tuple(map(int, axis_end)),
                color,
                6,
            )

        cv2.imwrite(
            str(output_dir / "calib_debug_img.png"),
            cv2.cvtColor(ref_frame, cv2.COLOR_RGB2BGR),
        )

    # Mark as done
    (output_dir / "done.txt").write_text("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("-d", "--device", help="Which device to run segmentation model on")
    args = parser.parse_args()
    calibrate_video(args.input_dir, args.device)
