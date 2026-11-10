"""Turn a GENMO result (hmr4d_results.pt) into the SMPL sequence, skeleton and foot contact of one player."""

import cv2
import numpy as np
import scipy.interpolate
import torch
from scipy.spatial.transform import Rotation as R

from rtt_utils.skeleton import select_target21_from_smpl24
from rtt_utils.smpl import (
    genmo_global_coords_to_table_coords,
    get_smpl_sole_indices,
    smpl_dicts_to_rotmats,
    smpl_forward_chunked,
    smpl_forward_pos,
)

_LEFT_ANKLE_IDX = 15  # COCO17
_RIGHT_ANKLE_IDX = 16  # COCO17
_SMPL_LEFT_ANKLE = 7
_SMPL_RIGHT_ANKLE = 8


def process_genmo_player(genmo_pred, g_wc, smpl_model, vitpose, H_ground):
    """
    Convert a loaded GENMO hmr4d_results dict into SMPL params, BVH skeleton,
    and foot contact for one player.

    Args:
        genmo_pred: dict from hmr4d_results.pt (must contain 'smpl_params_global')
        g_wc: world-to-camera transform (tt4d.g_wc)
        smpl_model: loaded SMPL model
        vitpose: (T, 17, 3) numpy array of COCO17 keypoints with confidence
        H_ground: (3, 3) ground homography matrix

    Returns:
        player_smpl: list of T dicts with rotation-matrix SMPL params
        player_bvh: np.ndarray (T, 21, 3)
        foot_contact: np.ndarray (T, 2) float32

    Raises:
        ValueError: if no valid vitpose ankle detections for offset correction
    """
    smpl_params = genmo_global_coords_to_table_coords(genmo_pred, g_wc, smpl_model=smpl_model)

    T = smpl_params["transl"].shape[0]
    body_pose_aa = smpl_params["body_pose"].reshape(T, 21, 3)
    betas = smpl_params["betas"]

    body_pose_rotmats = np.zeros((T, 23, 3, 3), dtype=np.float32)
    for t in range(T):
        body_pose_rotmats[t, :21] = R.from_rotvec(body_pose_aa[t]).as_matrix()
        body_pose_rotmats[t, 21:] = np.eye(3)

    global_orient_rotmats = R.from_rotvec(smpl_params["global_orient"]).as_matrix().astype(np.float32)
    transl = smpl_params["transl"]

    # Temp forward pass to get 3D joint positions for offset computation
    temp_player_smpl = [
        {
            "global_orient": global_orient_rotmats[t][np.newaxis],
            "body_pose": body_pose_rotmats[t],
            "betas": betas[t] if betas.ndim > 1 else betas,
            "transl": transl[t],
        }
        for t in range(T)
    ]
    go_t, bp_t, tr_t, bt_t = smpl_dicts_to_rotmats(temp_player_smpl)
    pos24_temp, _ = smpl_forward_pos(smpl_model, go_t, bp_t, tr_t, bt_t)

    # Homography-based translation offset from vitpose ankle positions
    H_ground_inv = np.linalg.inv(H_ground)
    offsets = []
    for t in range(min(T, len(vitpose))):
        left_ankle_3d = pos24_temp[t, _SMPL_LEFT_ANKLE]
        right_ankle_3d = pos24_temp[t, _SMPL_RIGHT_ANKLE]
        if left_ankle_3d[2] < right_ankle_3d[2]:
            ankle_2d = vitpose[t, _LEFT_ANKLE_IDX, :2]
            ankle_3d = left_ankle_3d
            conf = float(vitpose[t, _LEFT_ANKLE_IDX, 2])
        else:
            ankle_2d = vitpose[t, _RIGHT_ANKLE_IDX, :2]
            ankle_3d = right_ankle_3d
            conf = float(vitpose[t, _RIGHT_ANKLE_IDX, 2])

        if conf < 0.3:
            continue

        ground_pt = cv2.perspectiveTransform(ankle_2d[None, None, :].astype(np.float32), H_ground_inv)[0]
        offsets.append(np.array([ground_pt[0, 0], ground_pt[0, 1], 0.0]) - ankle_3d)

    if not offsets:
        raise ValueError("No valid vitpose ankle detections for homography offset correction")

    avg_offset = np.mean(offsets, axis=0)
    transl = transl + avg_offset

    # Build final player_smpl with corrected translations
    player_smpl = [
        {
            "global_orient": global_orient_rotmats[t][np.newaxis],
            "body_pose": body_pose_rotmats[t],
            "betas": betas[t] if betas.ndim > 1 else betas,
            "transl": transl[t],
        }
        for t in range(T)
    ]

    # BVH conversion
    go_mats, bp_mats, trans, betas_arr = smpl_dicts_to_rotmats(player_smpl)
    pos24, _ = smpl_forward_pos(smpl_model, go_mats, bp_mats, trans, betas_arr)
    player_bvh, _ = select_target21_from_smpl24(pos24)

    # Foot contact from static confidence logits
    _logits = genmo_pred.get("net_outputs", {}).get("static_conf_logits", None)
    foot_contact = np.full((T, 2), 0.5, dtype=np.float32)
    if _logits is not None:
        _logits_np = _logits.detach().cpu().numpy() if isinstance(_logits, torch.Tensor) else np.asarray(_logits)
        if _logits_np.ndim == 3:
            _logits_np = _logits_np[0]
        _probs = (1.0 / (1.0 + np.exp(-_logits_np.astype(np.float64)))).astype(np.float32)
        _fc = np.stack(
            [
                _probs[:, 0:2].max(axis=1),
                _probs[:, 2:4].max(axis=1),
            ],
            axis=1,
        )
        foot_contact = (_fc[:T] > 0.5).astype(np.float32)

    # Height adjustment via SMPL sole vertex z-positions
    SOLE_INDICES = get_smpl_sole_indices()
    N = SOLE_INDICES.shape[0] // 2
    _, vertices = smpl_forward_chunked(smpl_model, player_smpl)
    vertex_positions = vertices[:, SOLE_INDICES, :]

    left_z = np.mean(vertex_positions[:, :N, 2], axis=1, keepdims=True)
    right_z = np.mean(vertex_positions[:, N:, 2], axis=1, keepdims=True)
    feet_z_pos = np.concatenate([left_z, right_z], axis=1)
    feet_offset = np.where(foot_contact == 1, feet_z_pos, 0)
    dual_contact = np.all(feet_offset != 0, axis=1)
    z_offset = np.where(dual_contact, feet_offset.mean(axis=1), feet_offset.sum(axis=1))

    nz_indices = np.where(z_offset != 0)[0]
    if len(nz_indices) > 0:
        cs = scipy.interpolate.PchipInterpolator(nz_indices, z_offset[nz_indices])
        z_offset_interp = cs(np.arange(T))
        for i in range(T):
            player_smpl[i]["transl"][2] -= z_offset_interp[i]
            player_bvh[i][:, 2] -= z_offset_interp[i]

    return player_smpl, player_bvh, foot_contact
