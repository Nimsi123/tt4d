"""Constants and camera geometry shared by the ball lifting model and its runner (from Kienzle et al., WACV 2026)."""

import einops as eo
import numpy as np
import torch

from rtt_utils.constants import TABLE_HEIGHT, TABLE_LENGTH, TABLE_WIDTH

BALL_VISIBLE = 1
BALL_INVISIBLE = 0
KEYPOINT_VISIBLE = 1  # For Table Keypoints
KEYPOINT_INVISIBLE = 0  # For Table Keypoints

HEIGHT, WIDTH = 1440, 2560  # image resolution the model was trained at; inputs are rescaled to it
MAX_FPS = 500  # used to calculate the rotation angles in the rotary positional embedding -> Videos should be recorded with a lower fps

table_points = np.array(
    [
        [-TABLE_LENGTH / 2, TABLE_WIDTH / 2, TABLE_HEIGHT],  # 0 close left
        [-TABLE_LENGTH / 2, -TABLE_WIDTH / 2, TABLE_HEIGHT],  # 1 close right
        [0.0, TABLE_WIDTH / 2, TABLE_HEIGHT],  # 2 center left
        [0.0, -TABLE_WIDTH / 2, TABLE_HEIGHT],  # 3 center right
        [TABLE_LENGTH / 2, TABLE_WIDTH / 2, TABLE_HEIGHT],  # 4 far left
        [TABLE_LENGTH / 2, -TABLE_WIDTH / 2, TABLE_HEIGHT],  # 5 far right
        [0.0, TABLE_WIDTH / 2 + 0.1525, TABLE_HEIGHT],  # 6 net left bottom
        [0.0, -(TABLE_WIDTH / 2 + 0.1525), TABLE_HEIGHT],  # 7 net right bottom
        [0.0, 0.0, TABLE_HEIGHT],  # 8 net center bottom
        [0.0, TABLE_WIDTH / 2 + 0.1525, TABLE_HEIGHT + 0.1525],  # 9 net left top
        [0.0, -(TABLE_WIDTH / 2 + 0.1525), TABLE_HEIGHT + 0.1525],  # 10 net right top
        [-TABLE_LENGTH / 2, 0, TABLE_HEIGHT],  # 11 close center
        [TABLE_LENGTH / 2, 0, TABLE_HEIGHT],  # 12 far center
    ]
)


def cam2img(r_cam, Mints):
    '''Project a batch of 3D points to image coordinates.'''
    if len(r_cam.shape) == 1:
        if len(Mints.shape) == 2:
            r_img = eo.einsum(Mints[:3, :3], r_cam, 'i j, j -> i')
            r_img = r_img[:2] / r_img[2]
        else:
            raise ValueError('Shape not supported.')
    elif len(r_cam.shape) == 2:
        if len(Mints.shape) == 3:
            r_img = eo.einsum(Mints[:, :3, :3], r_cam, 'b i j, b j -> b i')
            r_img = r_img[:, :2] / r_img[:, 2:3]
        elif len(Mints.shape) == 2:
            r_img = eo.einsum(Mints[:3, :3], r_cam, 'i j, b j -> b i')
            r_img = r_img[:, :2] / r_img[:, 2:3]
        else:
            raise ValueError('Shape not supported.')
    elif len(r_cam.shape) == 3:
        if len(Mints.shape) == 3:
            r_img = eo.einsum(Mints[:, :3, :3], r_cam, 'b i j, b t j -> b t i')
            r_img = r_img[:, :, :2] / r_img[:, :, 2:3]
        elif len(Mints.shape) == 2:
            r_img = eo.einsum(Mints[:3, :3], r_cam, 'i j, b t j -> b t i')
            r_img = r_img[:, :, :2] / r_img[:, :, 2:3]
        else:
            raise ValueError('Shape not supported.')
    else:
        raise ValueError('Shape not supported.')
    return r_img


def world2cam(r_world, Mexts):
    '''Transform a batch of 3D points from world to camera coordinates.'''
    if len(r_world.shape) == 1:
        D = r_world.shape
        if len(Mexts.shape) == 2:
            r_world = concat(r_world, (D,))
            r_cam = eo.einsum(Mexts, r_world, 'i j, j -> i')
            r_cam = r_cam[:3] / r_cam[3]
        else:
            raise ValueError('Shape not supported.')
    elif len(r_world.shape) == 2:
        T, D = r_world.shape
        if len(Mexts.shape) == 3:
            r_world = concat(r_world, (T, D))
            r_cam = eo.einsum(Mexts, r_world, 'b i j, b j -> b i')
            r_cam = r_cam[:, :3] / r_cam[:, 3:4]
        elif len(Mexts.shape) == 2:
            r_world = concat(r_world, (T, D))
            r_cam = eo.einsum(Mexts, r_world, 'i j, b j -> b i')
            r_cam = r_cam[:, :3] / r_cam[:, 3:4]
        else:
            raise ValueError('Shape not supported.')
    elif len(r_world.shape) == 3:
        B, T, D = r_world.shape
        if len(Mexts.shape) == 3:
            r_world = concat(r_world, (B, T, D))
            r_cam = eo.einsum(Mexts, r_world, 'b i j, b t j -> b t i')
            r_cam = r_cam[:, :, :3] / r_cam[:, :, 3:4]
        elif len(Mexts.shape) == 2:
            r_world = concat(r_world, (B, T, D))
            r_cam = eo.einsum(Mexts, r_world, 'i j, b t j -> b t i')
            r_cam = r_cam[:, :, :3] / r_cam[:, :, 3:4]
        else:
            raise ValueError('Shape not supported.')
    else:
        raise ValueError('Shape not supported.')
    return r_cam


def concat(x, shape):
    """
    Concatenates a tensor `x` with a tensor of ones along the last dimension.
    Parameters:
    - x: Input tensor, either a numpy array or a PyTorch tensor.
    - shape: Desired shape of the tensor to concatenate (should match `x` except for the last dimension).
    Returns:
    - Concatenated tensor with an additional column of ones.
    """
    if isinstance(x, np.ndarray):
        ones = np.ones((*shape[:-1], 1))
        return np.concatenate([x, ones], axis=-1)
    elif isinstance(x, torch.Tensor):
        ones = torch.ones((*shape[:-1], 1), device=x.device)
        return torch.cat([x, ones], dim=-1)
    else:
        raise TypeError("Input must be either a numpy array or a torch tensor.")


def transform_rotationaxes(rotation, r_gt, mask=None):
    '''Rotates the axes such that the x axis is in flight direction, y axis in the worlds x-y plane.
    Uses per-frame flight direction (forward difference) to handle rallies with
    multiple shots going in different directions.

    Args:
        rotation (torch.tensor): rotation of the ball, shape (B, T, 3)
        r_gt (torch.tensor): trajectory (GT or predicted), shape (B, T, 3)
        mask (torch.tensor, optional): validity mask, shape (B, T), 1=valid, 0=invalid.
            Frames with mask=0 and their neighbors are excluded from velocity computation,
            and forward-filled from the last valid direction.
    '''
    assert len(r_gt.shape) == 3, "r_gt must have shape (B, T, 3)"
    B, T, _ = r_gt.shape
    device = r_gt.device

    # e_z is constant (vertical world axis) for all frames
    e_z = torch.zeros(B, T, 3, device=device, dtype=r_gt.dtype)
    e_z[..., 2] = 1.0

    # === Per-frame flight direction via forward difference ===
    velocity = torch.zeros(B, T, 3, device=device, dtype=r_gt.dtype)
    if T > 1:
        velocity[:, :-1] = r_gt[:, 1:] - r_gt[:, :-1]
        if mask is not None:
            valid_pair = mask[:, :-1].bool() & mask[:, 1:].bool()
            velocity[:, :-1] = velocity[:, :-1] * valid_pair.unsqueeze(-1)
        velocity[:, -1] = velocity[:, -2]

    # Project to XY plane (horizontal direction only)
    v_xy = velocity.clone()
    v_xy[..., 2] = 0.0

    # Normalize to get per-frame flight direction e_x
    v_norm = torch.linalg.norm(v_xy, dim=-1, keepdim=True)
    e_x = v_xy / torch.clamp(v_norm, min=1e-8)

    # === Forward-fill: propagate last valid e_x to zero-velocity frames ===
    has_valid_vel = v_norm.squeeze(-1) >= 1e-8
    if mask is not None:
        has_valid_vel = has_valid_vel & mask.bool()
    default_dir = torch.zeros(B, 3, device=device, dtype=r_gt.dtype)
    default_dir[:, 0] = 1.0
    last_valid = default_dir.clone()
    for t in range(T):
        valid_t = has_valid_vel[:, t]
        last_valid = torch.where(valid_t.unsqueeze(-1), e_x[:, t, :], last_valid)
        e_x[:, t, :] = torch.where((~valid_t).unsqueeze(-1), last_valid, e_x[:, t, :])

    # e_y = e_z x e_x (perpendicular in the XY plane)
    e_y = torch.cross(e_z, e_x, dim=-1)

    # Per-frame dot products
    w_0 = torch.einsum("bti,bti->bt", rotation, e_x)
    w_1 = torch.einsum("bti,bti->bt", rotation, e_y)
    w_2 = torch.einsum("bti,bti->bt", rotation, e_z)

    return torch.stack([w_0, w_1, w_2], dim=-1)
