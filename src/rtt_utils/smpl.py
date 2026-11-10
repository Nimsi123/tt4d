"""SMPL model loading and forward passes over the per-frame parameter dicts the pipeline stores."""

import os

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as _R

_DEFAULT_SMPL_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
_CHUNK = 64  # frames per SMPL forward pass in smpl_forward_chunked


def load_smpl_model(
    model_path: str = None,
    gender: str = "male",
    device: str = None,
    num_betas: int = 10,
):
    """Load the SMPL model from src/assets (or `model_path`) with smplx, in eval mode on `device`."""
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if model_path is None:
        model_path = _DEFAULT_SMPL_MODEL_DIR

    model = (
        smplx.create(
            model_path=model_path,
            model_type="smpl",
            gender=gender,
            num_betas=num_betas,
            create_global_orient=False,
            create_body_pose=False,
            create_betas=False,
            create_transl=False,
        )
        .to(device)
        .eval()
    )

    return model


def _squeeze_leading_ones(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    while a.ndim > 0 and a.shape[0] == 1:
        a = np.squeeze(a, axis=0)
    return a


def _to_rotmats(x) -> np.ndarray:
    """
    Accepts:
      - (...,3,3) rotation matrices
      - (...,3) axis-angles
      - (69,) flattened axis-angles (23*3)
      - Any of the above with leading singleton dims
    Returns:
      (...,3,3) float32 rotation matrices
    """
    a = _squeeze_leading_ones(np.asarray(x))

    # Already rotmats?
    if a.ndim >= 2 and a.shape[-2:] == (3, 3):
        return a.astype(np.float32)

    # Axis-angle cases
    if a.ndim == 1 and a.size % 3 == 0 or a.ndim >= 1 and a.shape[-1] == 3:  # e.g., (3,) or (69,)
        aa = a.reshape(-1, 3).astype(np.float64)
    else:
        raise ValueError(f"Unrecognized rotation shape {a.shape} for {type(x)}")

    mats = _R.from_rotvec(aa).as_matrix().astype(np.float32)  # (N,3,3)
    return mats.reshape(a.shape[:-1] + (3, 3))  # keep original leading dims


def _bodypose_to_mat23(x) -> np.ndarray:
    """Body pose → (23,3,3) rotmats (squeezes any leading singleton)."""
    M = _to_rotmats(x)
    # Expect exactly 23 joints after squeezing
    if M.shape == (23, 3, 3):
        return M
    raise ValueError(f"Expected (23,3,3) body pose after conversion, got {M.shape}")


def smpl_dicts_to_rotmats(smpl_seq) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    smpl_seq: list/np.array of dicts with keys:
      global_orient, body_pose, transl, (optional) betas.
    Returns:
      go_mats [T,1,3,3], bp_mats [T,23,3,3], trans [T,3], betas [1,K] or [T,K] or None
    """
    if isinstance(smpl_seq, np.ndarray):
        smpl_seq = smpl_seq.tolist()
    T = len(smpl_seq)
    go_mats = np.zeros((T, 1, 3, 3), dtype=np.float32)
    bp_mats = np.zeros((T, 23, 3, 3), dtype=np.float32)
    trans = np.zeros((T, 3), dtype=np.float32)
    betas_list: list[np.ndarray] = []

    for t, d in enumerate(smpl_seq):
        go = _to_rotmats(d["global_orient"])  # (3,3)
        bp = _bodypose_to_mat23(d["body_pose"])  # (23,3,3)
        tr = np.asarray(d["transl"], dtype=np.float32).reshape(3)

        go_mats[t, 0] = go
        bp_mats[t] = bp
        trans[t] = tr

        if "betas" in d and d["betas"] is not None:
            betas_list.append(np.asarray(d["betas"], dtype=np.float32).reshape(-1))

    betas = None
    if betas_list:
        betas = np.stack(betas_list, axis=0)  # [T,K]
        if np.allclose(betas, betas[0:1, :]):
            betas = betas[0:1, :]  # [1,K] if constant
    return go_mats, bp_mats, trans, betas


def get_smpl24_joints_from_vertices(model, out):
    """Regress the 24 SMPL joints from the output vertices with the model's J_regressor ([T,24,3] torch)."""
    V = out.vertices  # torch [T,V,3]
    Jreg = getattr(model, "J_regressor", None)
    if Jreg is None or np.sum(np.abs(Jreg.detach().cpu().numpy())) == 0:
        raise RuntimeError("Model has no J_regressor; cannot recover SMPL-24 joints.")
    if hasattr(Jreg, "to"):  # it’s usually a torch.FloatTensor / sparse
        Jreg = Jreg.to(V.device)
    # J = R * V  -> [T,24,3]
    J = torch.einsum('jv,tvc->tjc', Jreg, V)
    return J  # torch [T,24,3]


def smpl_forward_pos(model, go_mats, bp_mats, trans, betas=None):
    """
    model: SMPL module that accepts rotmats:
           global_orient [T,1,3,3], body_pose [T,23,3,3], transl [T,3], betas [T,K] or [1,K]
    Returns:
      pos24 [T,24,3] (meters), parents24 [24]
    """
    # Get device from the model (try buffers first, then parameters)
    # smplx models use buffers, not parameters
    device = next(iter(model.buffers()), next(iter(model.parameters()), None))
    device = device.device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    T = go_mats.shape[0]

    global_orient = torch.from_numpy(go_mats).float().to(device)  # [T,1,3,3]
    body_pose = torch.from_numpy(bp_mats).float().to(device)  # [T,23,3,3]
    transl = torch.from_numpy(trans).float().to(device)  # [T,3]

    betas_t = None
    if betas is not None:
        b = np.asarray(betas, dtype=np.float32)
        if b.ndim == 1:
            b = b[None, :]
        if b.shape[0] == 1:
            b = np.repeat(b, T, axis=0)
        betas_t = torch.from_numpy(b).float().to(device)

    with torch.no_grad():
        out = model(
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            betas=betas_t,
            pose2rot=False,  # We're already passing rotation matrices, not axis-angles
        )

    J = get_smpl24_joints_from_vertices(model, out)

    pos24 = J[:, :24, :].detach().cpu().numpy().astype(np.float64)

    # parents24 (try attribute; otherwise a common SMPL fallback)
    parents24 = None
    if hasattr(model, "parents"):
        parents24 = model.parents.detach().cpu().numpy().astype(np.int32)[:24]
    elif hasattr(model, "kintree_table"):
        kt = np.asarray(model.kintree_table)
        if kt.ndim == 2 and kt.shape[1] >= 24:
            # Many pickles store parents in row 0
            parents24 = kt[0, :24].astype(np.int32)
            # set root's parent = -1 if needed
            parents24[0] = -1
    if parents24 is None:
        # conservative fallback; you likely won't use smpl24 parents anyway if you view bvh21
        parents24 = np.array(
            [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19, 20, 21], dtype=np.int32
        )
    return pos24, parents24


def smpl_forward_chunked(smpl_model, player_smpl, device=None):
    """
    Run the SMPL model over a per-frame list of parameter dicts, 64 frames at a time.

    Poses may be rotation matrices or axis-angles (detected from the body_pose shape). Returns
    (joints [T,J,3], vertices [T,V,3]) as float32 numpy arrays.
    """
    if device is None:
        device = next(iter(smpl_model.buffers())).device
    T = len(player_smpl)
    joints, vertices = None, None
    for start in range(0, T, _CHUNK):
        end = min(start + _CHUNK, T)
        entries = player_smpl[start:end]

        def to_tensor(key, _entries=entries):
            return torch.from_numpy(np.stack([e[key] for e in _entries])).float().to(device)

        go, bp = to_tensor("global_orient"), to_tensor("body_pose")
        betas, transl = to_tensor("betas"), to_tensor("transl")
        is_rotmat = bp.ndim == 4 or (bp.ndim == 3 and bp.shape[-1] == 3)
        with torch.no_grad():
            out = smpl_model(
                global_orient=go.view(go.shape[0], -1, 3, 3) if is_rotmat else go.view(go.shape[0], 3),
                body_pose=bp.view(bp.shape[0], -1, 3, 3) if is_rotmat else bp.view(bp.shape[0], 69),
                betas=betas.view(betas.shape[0], -1),
                transl=transl.view(transl.shape[0], 3),
                pose2rot=not is_rotmat,
            )
        out_joints = out.joints.cpu().numpy().astype(np.float32)
        out_vertices = out.vertices.cpu().numpy().astype(np.float32)
        if joints is None:
            joints = np.zeros((T,) + out_joints.shape[1:], dtype=np.float32)
            vertices = np.zeros((T,) + out_vertices.shape[1:], dtype=np.float32)
        joints[start:end] = out_joints
        vertices[start:end] = out_vertices
    return joints, vertices


def get_smpl_sole_indices():
    """Indices of the 100 lowest vertices of each foot in the T-pose (left foot first), used as the soles."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl_model = load_smpl_model(model_path=None, gender="male", device=str(device))

    # --- STEP 1: Identify Sole Indices Geometrically ---
    # Run SMPL in default T-pose (zeros) to find the bottom vertices
    with torch.no_grad():
        # Canonical pose: all zeros
        canonical_output = smpl_model(
            global_orient=torch.zeros((1, 3)).to(device),
            body_pose=torch.zeros((1, 69)).to(device),
            betas=torch.zeros((1, 10)).to(device),
        )
        canonical_verts = canonical_output.vertices[0].cpu().numpy()

    # Separate vertices into left and right halves to ensure 10 points per foot
    # In SMPL, X > 0 is Left, X < 0 is Right (or vice versa depending on convention)
    mid_x = np.mean(canonical_verts[:, 0])
    left_mask = canonical_verts[:, 0] > mid_x
    right_mask = canonical_verts[:, 0] < mid_x

    # Get the 10 lowest vertices (min Y) for each side
    # If your model's 'up' is Z, change index [:, 1] to [:, 2]
    N = 100
    up_axis = 1
    l_sole_indices = np.where(left_mask)[0][np.argsort(canonical_verts[left_mask, up_axis])[:N]]
    r_sole_indices = np.where(right_mask)[0][np.argsort(canonical_verts[right_mask, up_axis])[:N]]

    return np.concatenate([l_sole_indices, r_sole_indices])


def genmo_global_coords_to_table_coords(genmo_pred, g_wc, smpl_model=None, device=None):
    """Transform GENMO smpl_params_global into the table tennis world frame.

    Mirrors compute_T_c2w + genmo_motion_apply_transform from visualize_smpl_global.py:
        T_c2w      = camera → GENMO-global (recovered from frame 0 of saved params)
        T_transform = g_wc @ inv(T_c2w)

    Args:
        genmo_pred: dict loaded from hmr4d_results.pt; must contain
                    'smpl_params_global' and 'smpl_params_incam'.
        g_wc:       (4, 4) float — camera → table-tennis world transform
                    (from load_calib / tt4d.g_wc).
        smpl_model: optional smplx.SMPL model; created on demand if None.
        device:     torch device string; auto-detected if None.

    Returns:
        dict with keys 'global_orient' (T, 3), 'body_pose' (T, 63),
        'transl' (T, 3), 'betas' (T, 10) — numpy float32, in the
        table-tennis world frame.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if smpl_model is None:
        smpl_model = load_smpl_model(device=device)

    sp_global = genmo_pred["smpl_params_global"]
    sp_incam = genmo_pred["smpl_params_incam"]

    def _np(v):
        return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)

    # --- Recover T_c2w from frame 0 (mirrors pp_static_joint_cam logic) ---
    R_gv = _R.from_rotvec(_np(sp_global["global_orient"])[0].reshape(3)).as_matrix().astype(np.float32)
    R_c = _R.from_rotvec(_np(sp_incam["global_orient"])[0].reshape(3)).as_matrix().astype(np.float32)
    R_c2w = R_gv @ R_c.T  # (3, 3)

    def _root_joint(sp) -> np.ndarray:
        """Run SMPL FK at frame 0 and return root joint position (3,)."""
        go_mat = _R.from_rotvec(_np(sp["global_orient"])[0].reshape(3)).as_matrix().astype(np.float32)
        go = torch.from_numpy(go_mat[None, None]).float().to(device)  # (1, 1, 3, 3)

        bp_aa = _np(sp["body_pose"])[0].reshape(-1, 3)  # (J, 3)
        bp_mats = np.zeros((23, 3, 3), dtype=np.float32)
        bp_mats[:] = np.eye(3)
        take = min(23, bp_aa.shape[0])
        bp_mats[:take] = _R.from_rotvec(bp_aa[:take]).as_matrix().astype(np.float32)
        bp = torch.from_numpy(bp_mats[None]).float().to(device)  # (1, 23, 3, 3)

        b = _np(sp["betas"])
        b = (b[0] if b.ndim == 2 else b).reshape(1, -1)
        t = torch.from_numpy(_np(sp["transl"])[0].reshape(1, 3)).float().to(device)
        with torch.no_grad():
            joints = smpl_model(
                global_orient=go,
                body_pose=bp,
                betas=torch.from_numpy(b).float().to(device),
                transl=t,
                pose2rot=False,
            ).joints
        return joints[0, 0].cpu().numpy()  # root joint (3,)

    t_c2w = _root_joint(sp_global) - R_c2w @ _root_joint(sp_incam)

    T_c2w = np.eye(4, dtype=np.float32)
    T_c2w[:3, :3] = R_c2w
    T_c2w[:3, 3] = t_c2w

    # --- Compose calibration with inverse of recovered T_c2w ---
    T_transform = (np.asarray(g_wc, dtype=np.float64) @ np.linalg.inv(T_c2w.astype(np.float64))).astype(np.float32)

    return genmo_motion_apply_transform(sp_global, T_transform)


def genmo_motion_apply_transform(smpl, transform):
    """Apply a rigid SE(3) transform to GENMO SMPL params.

    Args:
        smpl: dict with keys 'global_orient' (T, 3), 'transl' (T, 3),
              'body_pose', 'betas' (may be tensors or numpy).
        transform: (4, 4) homogeneous matrix [R | t]
            Applied as: orient_new = R @ orient, transl_new = R @ transl + t

    Returns:
        New dict with transformed 'global_orient' (T, 3) and 'transl' (T, 3);
        other keys copied unchanged.  All values are numpy float32.
    """
    R_mat, t_vec = transform[:3, :3], transform[:3, 3]

    def _to_np(v):
        return v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)

    orient = _to_np(smpl["global_orient"])  # (T, 3) axis-angle
    transl = _to_np(smpl["transl"])  # (T, 3)

    R_rot = _R.from_matrix(R_mat)
    new_orient = (R_rot * _R.from_rotvec(orient)).as_rotvec().astype(np.float32)
    new_transl = ((R_mat @ transl.T).T + t_vec).astype(np.float32)

    out = {}
    for k, v in smpl.items():
        if k == "global_orient":
            out[k] = new_orient
        elif k == "transl":
            out[k] = new_transl
        else:
            out[k] = _to_np(v)
    return out
