"""Camera calibration from the tt3d table calibrator, and projection between the world and the image."""

import os

import cv2
import numpy as np
import yaml

from rtt_utils.constants import TABLE_HEIGHT, TABLE_LENGTH, TABLE_WIDTH


def rotation_about_z(degrees: float) -> np.ndarray:
    """4x4 homogeneous rotation about the z axis."""
    θ = np.deg2rad(degrees)
    c, s = np.cos(θ), np.sin(θ)
    gz = np.array([[c, -s, 0.0, 0], [s, c, 0.0, 0], [0.0, 0.0, 1.0, 0], [0.0, 0.0, 0.0, 1.0]], dtype=float)
    return gz


def load_calib_extrinsics(gameplay_dir):
    """
    Read thomas_calibration/calib.yaml and return (rvecs, tvecs, focal_length) in the pipeline's world frame,
    without opening the clip's video (see load_calib, which adds the principal point).

    tt3d calibrates with the table surface at z=0 and its x axis along the table width; the pipeline puts the
    surface at z=TABLE_HEIGHT and the x axis along the table length, hence the translation and the 270 degree
    rotation about z. rvecs/tvecs are one-element lists (OpenCV convention).
    """
    with open(os.path.join(gameplay_dir, "thomas_calibration", "calib.yaml")) as f:
        calib = yaml.safe_load(f)  # keys: f, w, h, rvec, tvec

    if calib is None:
        return None, None, None

    rvec = np.asarray(calib["rvec"], dtype=np.float32)
    tvec = np.asarray(calib["tvec"], dtype=np.float32)
    focal_length = calib["f"]

    R_cw, _ = cv2.Rodrigues(rvec)
    T_cw = tvec
    R_wc = R_cw.T
    T_wc = -R_wc @ T_cw + np.array([0, 0, TABLE_HEIGHT], dtype=np.float32)

    g_wc = np.eye(4)
    g_wc[:3, :3] = R_wc
    g_wc[:3, 3] = T_wc

    g_wc = rotation_about_z(90 + 180) @ g_wc
    R_wc = g_wc[:3, :3]
    T_wc = g_wc[:3, 3:4]

    g_cw = np.linalg.inv(g_wc)
    R_cw_new = g_cw[:3, :3]
    T_cw_new = g_cw[:3, 3:4]  # (3,1)

    rvec_new, _ = cv2.Rodrigues(R_cw_new)

    rvecs = [rvec_new.astype(np.float32).reshape(3, 1)]
    tvecs = [T_cw_new.astype(np.float32).reshape(3, 1)]
    return rvecs, tvecs, focal_length


def load_calib(gameplay_dir):
    """
    Read thomas_calibration/calib.yaml and return (rvecs, tvecs, camera_matrix) in the pipeline's world frame.

    Same as load_calib_extrinsics, plus the intrinsics, which assume the principal point at the centre of
    the clip's first frame (this decodes one frame of the video).
    """
    clip_name = gameplay_dir.split("/")[-1]
    rvecs, tvecs, focal_length = load_calib_extrinsics(gameplay_dir)
    if rvecs is None:
        return None, None, None

    video_file = os.path.join(gameplay_dir, "video", f"{clip_name}.mp4")
    if not os.path.exists(video_file):
        # LATTE
        video_file = os.path.join(gameplay_dir, f"{clip_name}.mp4")
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {video_file}")
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Could not read a frame from {video_file}")
    height, width = frame.shape[:2]
    cap.release()

    camera_matrix = np.array([[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]], dtype=np.float32)

    return rvecs, tvecs, camera_matrix


def compute_table_diagonal_pixels(
    rvecs,
    tvecs,
    camera_matrix,
    dist_coeffs=None,
):
    """Length of the table-top diagonal in image pixels, used to normalize reprojection errors."""
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    # Use the first (and only) view
    rvec = np.asarray(rvecs[0], dtype=np.float32).reshape(3, 1)
    tvec = np.asarray(tvecs[0], dtype=np.float32).reshape(3, 1)

    # Table corners in world coordinates (centered table, z = table_height)
    L = TABLE_LENGTH
    W = TABLE_WIDTH
    H = TABLE_HEIGHT

    # Four corners on the tabletop:
    # (-L/2, -W/2), ( L/2, -W/2), ( L/2,  W/2), (-L/2,  W/2)
    object_points = np.array(
        [
            [-L / 2, -W / 2, H],
            [L / 2, -W / 2, H],
            [L / 2, W / 2, H],
            [-L / 2, W / 2, H],
        ],
        dtype=np.float32,
    )

    # Project into image
    img_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    img_points = img_points.reshape(-1, 2)

    # Take a diagonal between opposite corners, e.g. 0 ↔ 2
    p1 = img_points[0]
    p2 = img_points[2]
    diagonal_px = np.linalg.norm(p2 - p1)

    return diagonal_px


def project_3d_to_2d(camera_matrix, rvecs, tvecs, world_points):
    """Project (N, 3) world points to (N, 1, 2) image points with cv2.projectPoints and no distortion."""
    world_points = np.array(world_points, dtype=np.float32).reshape(-1, 1, 3)
    image_points, _ = cv2.projectPoints(world_points, rvecs[0], tvecs[0], camera_matrix, None)
    return image_points


def load_H_ground(rvecs, tvecs, camera_matrix):
    """Homography from ground-plane (x, y) in meters to image pixels, fitted on the table footprint at z=0."""
    L, B = TABLE_LENGTH, TABLE_WIDTH
    _3d_pts = [[-L / 2.0, -B / 2.0, 0.0], [-L / 2.0, B / 2.0, 0.0], [L / 2.0, -B / 2.0, 0.0], [L / 2.0, B / 2.0, 0.0]]
    _2d_pts = project_3d_to_2d(camera_matrix, rvecs, tvecs, np.array(_3d_pts))
    _2d_pts_homo = [[-L / 2.0, -B / 2.0], [-L / 2.0, B / 2.0], [L / 2.0, -B / 2.0], [L / 2.0, B / 2.0]]
    _2d_pts = np.array(_2d_pts, dtype=np.float32).reshape(-1, 2)
    _2d_pts_homo = np.array(_2d_pts_homo, dtype=np.float32).reshape(-1, 2)
    H_ground, status = cv2.findHomography(_2d_pts_homo, _2d_pts)

    return H_ground
