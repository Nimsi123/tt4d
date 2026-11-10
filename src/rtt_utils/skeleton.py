"""Joint layouts: SMPL 24 joints, the 21-joint Motive/BVH skeleton the TT4D stores, and SMPLH for retargeting."""

import numpy as np

from rtt_utils.smpl import load_smpl_model, smpl_dicts_to_rotmats, smpl_forward_pos

# --------------------------------------------
# 21-joint Motive/BVH skeleton (ORDERED_BVH_21)
# --------------------------------------------
MOTIVE21_NAMES = [
    "Hips",  # 0
    "Spine",  # 1
    "Spine1",  # 2
    "Neck",  # 3
    "Head",  # 4
    "LeftShoulder",  # 5
    "LeftArm",  # 6
    "LeftForeArm",  # 7
    "LeftHand",  # 8
    "RightShoulder",  # 9
    "RightArm",  # 10
    "RightForeArm",  # 11
    "RightHand",  # 12
    "LeftUpLeg",  # 13
    "LeftLeg",  # 14
    "LeftFoot",  # 15
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
]
MOTIVE21_PARENTS = np.array([-1, 0, 1, 2, 3, 2, 5, 6, 7, 2, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], dtype=np.int32)

# bones of the 21-joint skeleton as (parent, child) index pairs
SKELETON_21 = np.array(
    [
        # spine
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        # left arm
        [2, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        # right arm
        [2, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        # left leg
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        # right leg
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ],
    dtype=np.int32,
)
N_BONES_21 = len(SKELETON_21)


def bones_from_joints_21(j):  # j: (21,3) → (N_BONES_21,2,3)
    """Stack the endpoints of every bone of a (21, 3) skeleton into (N_BONES_21, 2, 3) line segments."""
    return np.stack([j[pair] for pair in SKELETON_21], axis=0)


# --------------------------------------------
# SMPL 24 joints → 21-joint skeleton
# --------------------------------------------
SMPL24_NAMES = [
    "Pelvis",
    "Left_hip",
    "Right_hip",
    "Spine1",
    "Left_knee",
    "Right_knee",
    "Spine2",
    "Left_ankle",
    "Right_ankle",
    "Spine3",
    "Left_foot",
    "Right_foot",
    "Neck",
    "Left_collar",
    "Right_collar",
    "Head",
    "Left_shoulder",
    "Right_shoulder",
    "Left_elbow",
    "Right_elbow",
    "Left_wrist",
    "Right_wrist",
    "Left_palm",
    "Right_palm",
]

MOTIVE21_TO_SMPL24 = {
    "Hips": "Pelvis",
    "Spine": "Spine1",
    "Spine1": "Spine2",
    "Neck": "Neck",
    "Head": "Head",
    "LeftShoulder": "Left_collar",
    "LeftArm": "Left_shoulder",
    "LeftForeArm": "Left_elbow",
    "LeftHand": "Left_wrist",
    "RightShoulder": "Right_collar",
    "RightArm": "Right_shoulder",
    "RightForeArm": "Right_elbow",
    "RightHand": "Right_wrist",
    "LeftUpLeg": "Left_hip",
    "LeftLeg": "Left_knee",
    "LeftFoot": "Left_ankle",
    "LeftToeBase": "Left_foot",
    "RightUpLeg": "Right_hip",
    "RightLeg": "Right_knee",
    "RightFoot": "Right_ankle",
    "RightToeBase": "Right_foot",
}


def select_target21_from_smpl24(pos24):
    """Pick the 21 Motive joints out of (T, 24, 3) SMPL joint positions; returns (pos21, MOTIVE21_PARENTS)."""
    smpl_index = {n: i for i, n in enumerate(SMPL24_NAMES)}
    idx21 = [smpl_index[MOTIVE21_TO_SMPL24[n]] for n in MOTIVE21_NAMES]
    return pos24[:, idx21, :], MOTIVE21_PARENTS.copy()


def define_skeleton_relative_to_root(seq):
    """Express joints 1.. relative to the root joint 0, which keeps its absolute position (inverse of define_skeleton_absolute)."""
    root_idx = 0
    root = seq[:, root_idx : root_idx + 1, :]
    new = seq - root
    new[:, root_idx, :] = root[:, 0, :]
    return new


def define_skeleton_absolute(seq):
    """Inverse of define_skeleton_relative_to_root."""
    root_idx = 0
    root = seq[:, root_idx : root_idx + 1, :]
    new = seq + root
    new[:, root_idx, :] = root[:, 0, :]
    return new


def geometric_adjustments_smpl_to_motive(skeleton: np.ndarray) -> np.ndarray:
    """Move the joints of an SMPL-derived (N, 21, 3) skeleton to where the Motive mocap skeleton places them."""
    # absolute (not relative to root)

    reshaped_skeleton = skeleton.copy()

    # Redefine joint 0 to be the midway point of the line segment (13, 17)
    reshaped_skeleton[:, 0, :] = (reshaped_skeleton[:, 13, :] + reshaped_skeleton[:, 17, :]) / 2

    # Bring joint 1 closer to joint 0
    beta = 0.4
    reshaped_skeleton[:, 1, :] = (
        beta * (reshaped_skeleton[:, 1, :] - reshaped_skeleton[:, 0, :]) + reshaped_skeleton[:, 0, :]
    )

    # Bring joint 3 closer to joint 2
    beta = 0.77
    reshaped_skeleton[:, 3, :] = (
        beta * (reshaped_skeleton[:, 3, :] - reshaped_skeleton[:, 2, :]) + reshaped_skeleton[:, 2, :]
    )

    # Redefine the 4 joint (the head joint) to be the extension of the vector (2, 3)
    # Add the length of the joint (3, 4) to the vector (2, 3)
    length = np.linalg.norm(reshaped_skeleton[:, 3, :] - reshaped_skeleton[:, 2, :], axis=1)  # (N,)
    length += np.linalg.norm(reshaped_skeleton[:, 4, :] - reshaped_skeleton[:, 3, :], axis=1)
    bone = reshaped_skeleton[:, 3, :] - reshaped_skeleton[:, 2, :]  # (N, 3)
    unit_bone = bone / np.linalg.norm(bone, axis=1)[:, None]
    reshaped_skeleton[:, 4, :] = length[:, None] * unit_bone + reshaped_skeleton[:, 2, :]

    # Bring joints (5, 6) and (9, 10) closer to the line segment defined by the line segment (2, 3)
    beta = 0.3
    idx = [5, 6, 9, 10]
    P = reshaped_skeleton[:, idx, :]  # (N, 4, 3)
    B = reshaped_skeleton[:, 2:3, :]  # (N, 1, 3)
    U = unit_bone[:, None, :]  # (N, 1, 3)
    T = ((P - B) * U).sum(-1, keepdims=True)  # (N, 4, 1)
    PROJ = B + T * U  # (N, 4, 3)
    reshaped_skeleton[:, idx, :] = P + beta * (PROJ - P)

    # The skeleton should not be on the floor
    ground_penetration = np.min(reshaped_skeleton[:, :, 2], axis=1)[:, None]  # (N, 1)
    reshaped_skeleton[:, :, 2] -= ground_penetration

    return reshaped_skeleton


# --------------------------------------------
# SMPL 24 joints → SMPLH 52 joints for retargeting
# --------------------------------------------
# Mapping from SMPLH joint index to SMPL24 joint index
# Finger joints map to wrist since SMPL24 doesn't have fingers
SMPLH_TO_SMPL24_INDICES = {
    0: 0,  # Pelvis -> Pelvis
    1: 1,  # L_Hip -> Left_hip
    2: 4,  # L_Knee -> Left_knee
    3: 7,  # L_Ankle -> Left_ankle
    4: 10,  # L_Toe -> Left_foot
    5: 2,  # R_Hip -> Right_hip
    6: 5,  # R_Knee -> Right_knee
    7: 8,  # R_Ankle -> Right_ankle
    8: 11,  # R_Toe -> Right_foot
    9: 3,  # Torso -> Spine1
    10: 6,  # Spine -> Spine2
    11: 9,  # Chest -> Spine3
    12: 12,  # Neck -> Neck
    13: 15,  # Head -> Head
    14: 13,  # L_Thorax -> Left_collar
    15: 16,  # L_Shoulder -> Left_shoulder
    16: 18,  # L_Elbow -> Left_elbow
    17: 20,  # L_Wrist -> Left_wrist
    # Left hand fingers (18-32) -> use L_Wrist position (index 20)
    18: 20,
    19: 20,
    20: 20,  # L_Index
    21: 20,
    22: 20,
    23: 20,  # L_Middle
    24: 20,
    25: 20,
    26: 20,  # L_Pinky
    27: 20,
    28: 20,
    29: 20,  # L_Ring
    30: 20,
    31: 20,
    32: 20,  # L_Thumb
    33: 14,  # R_Thorax -> Right_collar
    34: 17,  # R_Shoulder -> Right_shoulder
    35: 19,  # R_Elbow -> Right_elbow
    36: 21,  # R_Wrist -> Right_wrist
    # Right hand fingers (37-51) -> use R_Wrist position (index 21)
    37: 21,
    38: 21,
    39: 21,  # R_Index
    40: 21,
    41: 21,
    42: 21,  # R_Middle
    43: 21,
    44: 21,
    45: 21,  # R_Pinky
    46: 21,
    47: 21,
    48: 21,  # R_Ring
    49: 21,
    50: 21,
    51: 21,  # R_Thumb
}


def smpl24_to_smplh52(pos24: np.ndarray) -> np.ndarray:
    """Expand (T, 24, 3) SMPL joints to the (T, 52, 3) SMPLH layout; finger joints copy their wrist."""
    T = pos24.shape[0]
    pos52 = np.zeros((T, 52, 3), dtype=pos24.dtype)

    for smplh_idx, smpl24_idx in SMPLH_TO_SMPL24_INDICES.items():
        pos52[:, smplh_idx, :] = pos24[:, smpl24_idx, :]

    return pos52


def smpl_to_smplh(smpl_seq, smpl_model=None):
    """Run SMPL on a sequence of parameter dicts and return (T, 52, 3) SMPLH joint positions for retargeting."""
    if smpl_model is None:
        smpl_model = load_smpl_model()

    # Convert SMPL dicts to rotation matrices
    go_mats, bp_mats, trans, betas = smpl_dicts_to_rotmats(smpl_seq)

    # Run SMPL forward kinematics to get 24-joint world positions
    pos24, _ = smpl_forward_pos(smpl_model, go_mats, bp_mats, trans, betas)

    # Convert 24-joint SMPL to 52-joint SMPLH format
    pos_smplh = smpl24_to_smplh52(pos24)

    return pos_smplh
