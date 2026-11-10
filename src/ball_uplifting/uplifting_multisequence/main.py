"""
Parallel script to process TT4D objects with ball uplifting (interpolated model).
Loads pre-created TT4D pkl files, runs the uplifting model, and saves processed results.
Stores ball_3d/spin_3d (interp, all frames).

Usage (from src/ball_uplifting):
    # one worker on each of four GPUs
    python -m uplifting_multisequence.main --resource-ids 0 1 2 3

    # the same, split across 4 machines (this is machine 0)
    python -m uplifting_multisequence.main --resource-ids 0 1 2 3 --num-workers 4 --worker-split 0
"""

import os
import shutil

import einops as eo
import numpy as np
import torch
import tyro

from ball_uplifting.uplifting_multisequence.helper import (
    BALL_INVISIBLE,
    BALL_VISIBLE,
    HEIGHT,
    WIDTH,
    cam2img,
    table_points,
    transform_rotationaxes,
    world2cam,
)
from ball_uplifting.uplifting_multisequence.model import get_model
from ball_uplifting.uplifting_multisequence.transformations import get_transforms
from rtt_utils.constants import MAX_NEURAL_REPROJECTION_THRESHOLD, MAX_RMSE_THRESHOLD
from rtt_utils.parallel import ParallelConfig, run_parallel
from rtt_utils.paths import get_home_path
from ttg.types.tt4d import TT4D

weights_path = f'{get_home_path()}/src/ball_uplifting/model'
model_withinterpolation_path = f'{weights_path}/model_empiricalAugmentation_spincorrection.pt'

SEQUENCE_LEN = 1000


def load_model(model_path):
    '''
    Load the uplifting model from the given path.
    Args:
        model_path (str): Path to the saved model.
    Returns:
        model (torch.nn.Module): Loaded uplifting model.
        transform (callable): Transformation function for input images.
        transform_mode (str): Spin transformation mode used during training ('global' or 'local').
        interpolate_missing (bool): Whether the model was trained with interpolation of missing detections.
    '''
    loaded_dict = torch.load(model_path, weights_only=False)
    model_name = loaded_dict['additional_info']['name']
    model_size = loaded_dict['additional_info']['size']
    tabletoken_mode = loaded_dict['additional_info']['tabletoken_mode']
    time_rotation = loaded_dict['additional_info']['time_rotation']
    transform_mode = loaded_dict['additional_info']['transform_mode']
    randmiss_prob = loaded_dict['additional_info']['randmiss_prob']
    try:
        interpolate_missing = loaded_dict['additional_info']['interpolate_missing']
        strict = True
    except KeyError:
        interpolate_missing = False
        strict = False
        print('Old model architecture is loaded -> no missing_token saved')
    uplifting_model = get_model(
        model_name,
        size=model_size,
        mode=tabletoken_mode,
        time_rotation=time_rotation,
        interpolate_missing=interpolate_missing,
    )
    uplifting_model.load_state_dict(loaded_dict['model_state_dict'], strict=strict)
    uplifting_model.eval()
    print(
        f'Loaded Uplifting model: {model_name} with size {model_size}, tabletoken_mode: {tabletoken_mode}, time_rotation: {time_rotation}, transform_mode: {transform_mode}'
    )
    print(f'Noise settings during training - randmiss_prob: {randmiss_prob}')
    # Use the standard transforms for evaluation because data comes from the dataloader, not the detection models
    transform = get_transforms()
    return uplifting_model, transform, transform_mode, interpolate_missing


def lift_ball(r_img, table_img, mask, times, M_int, M_ext, model, transform_mode, interpolate_missing, device):
    """Run forward pass for a single uplifting model.

    Returns:
        (pred_position_np, pred_rotation_np)
    """
    # Add batch dimension and move to device
    r_img = r_img.unsqueeze(0).to(device)
    table_img = table_img.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    times = times.unsqueeze(0).to(device)

    # Collapse time dimension for table coordinates (static camera)
    table_img_input = torch.median(table_img, dim=1).values
    valid_len = int(mask.squeeze(0).cpu().sum().item())

    with torch.no_grad():
        pred_rotation, pred_position = model(r_img, table_img_input, mask, times)

    # Transform prediction into local coordinate system
    if transform_mode == 'global':
        pred_rotation_local = transform_rotationaxes(pred_rotation, pred_position.clone(), mask)
    else:
        pred_rotation_local = pred_rotation
    pred_rotation_local = pred_rotation_local / (2 * np.pi)  # rad/s to Hz

    pred_rotation_np = pred_rotation_local.squeeze(0).cpu().numpy()[:valid_len]
    pred_position_np = pred_position.squeeze(0).cpu().numpy()[:valid_len]

    # When interpolate_missing=False, the model packs visible frames to the left.
    # Unpack predictions back to original frame order using sort_indices.
    if not interpolate_missing:
        visibilities = r_img[0, :valid_len, 2]
        keep_mask = visibilities == BALL_VISIBLE
        _, sort_indices = torch.sort(keep_mask.int().cpu(), dim=0, descending=True, stable=True)
        sort_indices = sort_indices.numpy()
        pred_position_unpacked = np.zeros_like(pred_position_np)
        pred_rotation_unpacked = np.zeros_like(pred_rotation_np)
        pred_position_unpacked[sort_indices] = pred_position_np
        pred_rotation_unpacked[sort_indices] = pred_rotation_np
        pred_position_np = pred_position_unpacked
        pred_rotation_np = pred_rotation_unpacked

    return pred_position_np, pred_rotation_np


def prepare_ball_lift_input(tt4d, transforms):
    """Prepare model inputs from a TT4D object.

    Replaces the lifting model's own preprocessing: instead of loading calibration and
    ball data from the filesystem, this pulls everything from the TT4D pkl.

    The calibration is converted to the lifting model's coordinate frame (table-height
    origin at z=0.76 m, with a 90-degree axis swap) to match what the
    uplifting model was trained on.

    Returns:
        (r_img, table_img, mask, times, M_int, M_ext) as torch tensors.
    """
    K = np.array(tt4d.camera_matrix, dtype=np.float32).copy()

    # TT4D's g_wc is in the rtt_utils frame, which applies BOTH a table-height
    # offset (z=0.76 m) AND a 270° z-rotation on top of the raw calibration
    # (see load_calib: rotation_about_z(90+180) @ g_wc).
    #
    # The uplifting model's preprocessing expects its own frame,
    # which has the table-height offset but NO z-rotation.
    # Undo the rotation: g_wc_uplift = Rz(-(90+180)) @ g_wc_rtt
    theta = np.deg2rad(-(90 + 180))
    c, s = np.cos(theta), np.sin(theta)
    R_z_inv = np.array(
        [
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )

    g_wc_uplift = R_z_inv @ tt4d.g_wc
    g_cw_uplift = np.linalg.inv(g_wc_uplift)
    R = g_cw_uplift[:3, :3].astype(np.float32)
    t = g_cw_uplift[:3, 3:4].astype(np.float32)  # (3, 1)

    # Extract 2D ball coordinates and visibility from TT4D
    r_img = tt4d.ball_2d[:, 2:4].astype(np.float32).copy()  # (T, 2)
    vis = tt4d.ball_2d[:, 1].copy()
    times = (np.arange(len(r_img)) / tt4d.fps).astype(np.float32)

    # Derive original video resolution from camera principal point
    old_W = int(round(K[0, 2] * 2))
    old_H = int(round(K[1, 2] * 2))

    # Build extrinsic matrix with the lifting model's 90-degree axis swap
    K, R, t = K.copy(), R.copy(), t.copy()
    M_ext = np.eye(4, dtype=np.float32)
    tmp = R.copy()
    R[:, 0], R[:, 1] = tmp[:, 1], -tmp[:, 0]
    M_ext[:3, :3] = R
    M_ext[:3, 3] = t.squeeze(1)
    M_ext = eo.repeat(M_ext, 'i j -> t i j', t=r_img.shape[0])  # (T, 4, 4)
    M_int = eo.repeat(K, 'i j -> t i j', t=r_img.shape[0])  # (T, 3, 3)

    # Project table reference points into image coordinates
    table_world = np.array(table_points, dtype=np.float32)
    tw = eo.repeat(table_world, 'n j -> t n j', t=r_img.shape[0])
    table_img = cam2img(world2cam(tw, M_ext), M_int)

    # Pad / crop sequence to SEQUENCE_LEN
    T, _ = r_img.shape
    mask = np.ones(SEQUENCE_LEN, dtype=np.float32)
    if T < SEQUENCE_LEN:
        mask[T:] = 0

    max_t = min(T, SEQUENCE_LEN)
    tmp = np.zeros((SEQUENCE_LEN, 3), dtype=np.float32)
    tmp[:max_t, :2] = r_img[:max_t]
    tmp[:max_t, 2] = np.where(vis[:max_t] == 1, BALL_VISIBLE, BALL_INVISIBLE)
    r_img = tmp
    tmp = np.zeros((SEQUENCE_LEN,), dtype=np.float32)
    tmp[:max_t] = times[:max_t]
    times = tmp

    # Rescale ball coords to evaluation resolution (HEIGHT x WIDTH)
    r_img[:, 0] = (r_img[:, 0] + 0.5) * WIDTH / old_W - 0.5
    r_img[:, 1] = (r_img[:, 1] + 0.5) * HEIGHT / old_H - 0.5
    table_img[:, :, 0] = (table_img[:, :, 0] + 0.5) * WIDTH / old_W - 0.5
    table_img[:, :, 1] = (table_img[:, :, 1] + 0.5) * HEIGHT / old_H - 0.5
    M_int[:, 0, 0] = M_int[:, 0, 0] * WIDTH / old_W
    M_int[:, 1, 1] = M_int[:, 1, 1] * HEIGHT / old_H
    M_int[:, 0, 2] = M_int[:, 0, 2] * WIDTH / old_W
    M_int[:, 1, 2] = M_int[:, 1, 2] * HEIGHT / old_H

    # Apply test-time transforms
    data = {
        'r_img': r_img,
        'times': times,
        'mask': mask,
        'table_img': table_img,
        'Mint': M_int,
        'Mext': M_ext,
    }
    if transforms is not None:
        data = transforms(data)
    r_img, table_img, mask = data['r_img'], data['table_img'], data['mask']
    times, M_int, M_ext = data['times'], data['Mint'], data['Mext']

    dtype = torch.float32
    r_img = torch.tensor(r_img, dtype=dtype)
    table_img = torch.tensor(table_img, dtype=dtype)
    mask = torch.tensor(mask, dtype=dtype)
    times = torch.tensor(times, dtype=dtype)
    M_int = torch.tensor(M_int, dtype=dtype)
    M_ext = torch.tensor(M_ext, dtype=dtype)

    return r_img, table_img, mask, times, M_int, M_ext


def run_task(task, smpl_model, models, transforms, device):
    """Lift one clip's 2D ball track to 3D positions and spin, run TT4D.process and determine_validity, and save.

    Writes dani_ball/3d_{ball,spin}_interp.npy and nima_recons/<clip>_tt4d_processed.pkl (also for invalid clips,
    so the failure reason is kept). `models` maps a suffix to (model, transform_mode, interpolate_missing).
    """
    gameplay_dir, clip_name, pkl_path, pkl_processed_path = task

    if not os.path.exists(pkl_path):
        return
    if os.path.exists(pkl_processed_path):
        os.remove(pkl_processed_path)

    # load the partial TT4D object from src/mono_view_recons/runners/partial_recons.py
    tt4d = TT4D.load(pkl_path)
    tt4d.gameplay_dir = gameplay_dir

    if tt4d.invalid:
        print(f"[{clip_name}] Invalid: {tt4d.failure_reason}")
        tt4d.save(pkl_processed_path)
        return

    ball_save_dir = os.path.join(gameplay_dir, "dani_ball")
    if os.path.exists(ball_save_dir):
        shutil.rmtree(ball_save_dir)
    os.makedirs(ball_save_dir, exist_ok=True)

    if tt4d.invalid:
        print(f"[{clip_name}] Invalid after processing: {tt4d.failure_reason}")
        tt4d.save(pkl_processed_path)
        return
    r_img, table_img, mask, times, M_int, M_ext = prepare_ball_lift_input(tt4d, transforms)
    model, transform_mode, interpolate_missing = models["interp"]
    pred_position_np, pred_rotation_np = lift_ball(
        r_img,
        table_img,
        mask,
        times,
        M_int,
        M_ext,
        model,
        transform_mode,
        interpolate_missing,
        device,
    )
    tt4d.ball_3d = pred_position_np
    tt4d.spin_3d = pred_rotation_np

    suffix = "interp"
    np.save(os.path.join(ball_save_dir, f"3d_ball_{suffix}.npy"), pred_position_np)
    np.save(os.path.join(ball_save_dir, f"3d_spin_{suffix}.npy"), pred_rotation_np)
    tt4d.process(smpl_model)

    tt4d.determine_validity(MAX_RMSE_THRESHOLD, MAX_NEURAL_REPROJECTION_THRESHOLD, stats=True)

    if tt4d.invalid:
        print(f"[{clip_name}] Invalid after processing: {tt4d.failure_reason}")
        tt4d.save(pkl_processed_path)
        return

    tt4d.save(pkl_processed_path)
    print(f"Saved: {pkl_processed_path}")


def load_models(gpu_id):
    """Load SMPL and the uplifting model once per worker; returns (smpl_model, models dict, transforms)."""
    device = "cuda:0"
    smpl_model = TT4D._get_smpl(device=device)

    model_interp, transforms, tm_interp, interp_miss_interp = load_model(model_withinterpolation_path)
    model_interp = model_interp.to(device)

    models = {
        "interp": (model_interp, tm_interp, interp_miss_interp),
    }
    return smpl_model, models, transforms


def run_task_with_models(task, state):
    smpl_model, models, transforms = state
    run_task(task, smpl_model, models, transforms, "cuda:0")


def discover_tasks(root_dir, batch_indices):
    """Discover all gameplay directories that need processing."""
    tasks = []

    for i in batch_indices:
        batch_dir = os.path.join(root_dir, f"batch_clipped_long_{i}")
        if not os.path.exists(batch_dir):
            continue
        print(f"Going through {batch_dir}.")

        for clip_name in os.listdir(batch_dir):
            gameplay_dir = os.path.join(batch_dir, clip_name)

            pkl_path = os.path.join(gameplay_dir, "nima_recons", f"{clip_name}_tt4d.pkl")
            pkl_processed_path = os.path.join(gameplay_dir, "nima_recons", f"{clip_name}_tt4d_processed.pkl")

            tasks.append((gameplay_dir, clip_name, pkl_path, pkl_processed_path))

    tasks = sorted(tasks, key=lambda x: x[0])
    print(f"Total tasks: {len(tasks)}")
    return tasks


if __name__ == "__main__":
    "python -m uplifting_multisequence.main  --resource_ids 0 1 2 3 --workers_per_resource 1"

    config = tyro.cli(ParallelConfig)
    tasks = discover_tasks(config.get_root_dir(), config.get_batch_indices())
    tasks = config.slice_tasks(tasks)
    run_parallel(tasks, run_task_with_models, config, setup=load_models)
