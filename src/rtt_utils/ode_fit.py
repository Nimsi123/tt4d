import pickle
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from rtt_utils.physics_constants import COR, KD, KM, MU, M, R

G = np.array([0.0, 0.0, -9.81], dtype=np.float64)
ALPHA_ROLL_THRESHOLD = 0.4
EPS = 1e-12
TABLE_Z = 0.78  # meters
# TABLE_Z = 0.8  # meters


try:
    from numba import njit

    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

    def njit(*args, **kwargs):
        def deco(f):
            return f

        return deco


# Physics
@njit(fastmath=True)
def _norm3(v):
    return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5


@njit(fastmath=True)
def _cross(a, b, out):
    out[0] = a[1] * b[2] - a[2] * b[1]
    out[1] = a[2] * b[0] - a[0] * b[2]
    out[2] = a[0] * b[1] - a[1] * b[0]


@njit(fastmath=True)
def ball_accel_jit(v, w, out_acc):
    # a = g + ( -KD*|v|*v + KM*(w x v) ) / M
    speed = _norm3(v)
    tmp = np.empty(3, dtype=np.float64)
    _cross(w, v, tmp)
    out_acc[0] = G[0] + (-KD * speed * v[0] + KM * tmp[0]) / M
    out_acc[1] = G[1] + (-KD * speed * v[1] + KM * tmp[1]) / M
    out_acc[2] = G[2] + (-KD * speed * v[2] + KM * tmp[2]) / M


@njit(fastmath=True)
def rk4_step_jit(p, v, w, dt, p_next, v_next):
    a1 = np.empty(3, dtype=np.float64)
    a2 = np.empty(3, dtype=np.float64)
    a3 = np.empty(3, dtype=np.float64)
    a4 = np.empty(3, dtype=np.float64)

    # k1
    ball_accel_jit(v, w, a1)

    # k2 inputs
    v2 = np.empty(3, dtype=np.float64)
    p2 = np.empty(3, dtype=np.float64)
    v2[0] = v[0] + 0.5 * dt * a1[0]
    v2[1] = v[1] + 0.5 * dt * a1[1]
    v2[2] = v[2] + 0.5 * dt * a1[2]
    p2[0] = p[0] + 0.5 * dt * v[0]
    p2[1] = p[1] + 0.5 * dt * v[1]
    p2[2] = p[2] + 0.5 * dt * v[2]
    ball_accel_jit(v2, w, a2)

    # k3 inputs
    v3 = np.empty(3, dtype=np.float64)
    p3 = np.empty(3, dtype=np.float64)
    v3[0] = v[0] + 0.5 * dt * a2[0]
    v3[1] = v[1] + 0.5 * dt * a2[1]
    v3[2] = v[2] + 0.5 * dt * a2[2]
    p3[0] = p[0] + 0.5 * dt * v2[0]
    p3[1] = p[1] + 0.5 * dt * v2[1]
    p3[2] = p[2] + 0.5 * dt * v2[2]
    ball_accel_jit(v3, w, a3)

    # k4 inputs
    v4 = np.empty(3, dtype=np.float64)
    p4 = np.empty(3, dtype=np.float64)
    v4[0] = v[0] + dt * a3[0]
    v4[1] = v[1] + dt * a3[1]
    v4[2] = v[2] + dt * a3[2]
    p4[0] = p[0] + dt * v3[0]
    p4[1] = p[1] + dt * v3[1]
    p4[2] = p[2] + dt * v3[2]
    ball_accel_jit(v4, w, a4)

    # combine
    p_next[0] = p[0] + (dt / 6.0) * (v[0] + 2.0 * v2[0] + 2.0 * v3[0] + v4[0])
    p_next[1] = p[1] + (dt / 6.0) * (v[1] + 2.0 * v2[1] + 2.0 * v3[1] + v4[1])
    p_next[2] = p[2] + (dt / 6.0) * (v[2] + 2.0 * v2[2] + 2.0 * v3[2] + v4[2])

    v_next[0] = v[0] + (dt / 6.0) * (a1[0] + 2.0 * a2[0] + 2.0 * a3[0] + a4[0])
    v_next[1] = v[1] + (dt / 6.0) * (a1[1] + 2.0 * a2[1] + 2.0 * a3[1] + a4[1])
    v_next[2] = v[2] + (dt / 6.0) * (a1[2] + 2.0 * a2[2] + 2.0 * a3[2] + a4[2])


@njit(fastmath=True)
def surface_speed_xy_jit(v, w):
    sx = v[0] - R * w[1]
    sy = v[1] + R * w[0]
    return (sx * sx + sy * sy) ** 0.5


@njit(fastmath=True)
def alpha_value_jit(v):
    sv = surface_speed_xy_jit(v, np.zeros(3))  # placeholder to satisfy signature? No — we'll pass w separately.
    return sv  # dead code — we rewrite properly below


@njit(fastmath=True)
def alpha_value_full(v, w):
    sv = surface_speed_xy_jit(v, w)
    if sv < EPS:
        return 0.0
    vz = abs(v[2])
    return MU * (1.0 + COR) * vz / sv


@njit(fastmath=True)
def bounce_model_jit(v_in, w_in, v_out, w_out):
    a = alpha_value_full(v_in, w_in)

    if a > ALPHA_ROLL_THRESHOLD:
        # roll matrices
        # v' = A v + B w ; w' = C v + D w
        # hard-code 3x3 multiplies to avoid small temporaries
        # A_roll = diag(0.6,0.6,-COR)
        v_out[0] = 0.6 * v_in[0] + 0.4 * R * w_in[1]
        v_out[1] = 0.6 * v_in[1] - 0.4 * R * w_in[0]
        v_out[2] = -COR * v_in[2]

        w_out[0] = (-0.6 / R) * v_in[1] + 0.4 * w_in[0]
        w_out[1] = (0.6 / R) * v_in[0] + 0.4 * w_in[1]
        w_out[2] = w_in[2]
    else:
        # slide matrices
        one_minus_a = 1.0 - a
        one_minus_1p5a = 1.0 - 1.5 * a

        v_out[0] = one_minus_a * v_in[0] + a * R * w_in[1]
        v_out[1] = one_minus_a * v_in[1] - a * R * w_in[0]
        v_out[2] = -COR * v_in[2]

        w_out[0] = (-1.5 * a / R) * v_in[1] + one_minus_1p5a * w_in[0]
        w_out[1] = (1.5 * a / R) * v_in[0] + one_minus_1p5a * w_in[1]
        w_out[2] = w_in[2]


@njit(fastmath=True)
def simulate_with_bounces_jit(p0, v0, w0, t, table_z, max_bounces, substep, P_out):
    n = t.shape[0]
    p = np.empty(3, dtype=np.float64)
    v = np.empty(3, dtype=np.float64)
    w = np.empty(3, dtype=np.float64)
    p[0], p[1], p[2] = p0[0], p0[1], p0[2]
    v[0], v[1], v[2] = v0[0], v0[1], v0[2]
    w[0], w[1], w[2] = w0[0], w0[1], w0[2]

    P_out[0, 0] = p[0]
    P_out[0, 1] = p[1]
    P_out[0, 2] = p[2]

    bounces = 0
    t_prev = t[0]
    p_old = np.empty(3, dtype=np.float64)
    v_old = np.empty(3, dtype=np.float64)
    p_next = np.empty(3, dtype=np.float64)
    v_next = np.empty(3, dtype=np.float64)

    for i in range(1, n):
        dt_total = t[i] - t_prev
        if dt_total == 0.0:
            P_out[i, 0] = p[0]
            P_out[i, 1] = p[1]
            P_out[i, 2] = p[2]
            continue

        # integer number of substeps
        steps = 1
        if dt_total > 0.0:
            steps = int(np.ceil(dt_total / substep))
            if steps < 1:
                steps = 1
        elif dt_total < 0.0:
            steps = int(np.ceil((-dt_total) / substep))
            if steps < 1:
                steps = 1
        h = dt_total / steps

        for _ in range(steps):
            # store old state
            p_old[0], p_old[1], p_old[2] = p[0], p[1], p[2]
            v_old[0], v_old[1], v_old[2] = v[0], v[1], v[2]

            rk4_step_jit(p, v, w, h, p_next, v_next)
            p[0], p[1], p[2] = p_next[0], p_next[1], p_next[2]
            v[0], v[1], v[2] = v_next[0], v_next[1], v_next[2]

            if bounces < max_bounces:
                crossed = (p_old[2] > table_z) and (p[2] <= table_z) and (v_old[2] < 0.0)
                if crossed:
                    denom = p[2] - p_old[2]
                    s = 0.0
                    if abs(denom) >= 1e-12:
                        s = (table_z - p_old[2]) / denom
                        if s < 0.0:
                            s = 0.0
                        elif s > 1.0:
                            s = 1.0

                    # partial step to hit
                    rk4_step_jit(p_old, v_old, w, h * s, p_next, v_next)
                    p_next[2] = table_z

                    # bounce
                    v_after = np.empty(3, dtype=np.float64)
                    w_after = np.empty(3, dtype=np.float64)
                    bounce_model_jit(v_next, w, v_after, w_after)

                    p[0], p[1], p[2] = p_next[0], p_next[1], p_next[2]
                    v[0], v[1], v[2] = v_after[0], v_after[1], v_after[2]
                    w[0], w[1], w[2] = w_after[0], w_after[1], w_after[2]
                    bounces += 1

                    # remainder of the step
                    h_rem = h * (1.0 - s)
                    if h_rem != 0.0:
                        rk4_step_jit(p, v, w, h_rem, p_next, v_next)
                        p[0], p[1], p[2] = p_next[0], p_next[1], p_next[2]
                        v[0], v[1], v[2] = v_next[0], v_next[1], v_next[2]

        P_out[i, 0] = p[0]
        P_out[i, 1] = p[1]
        P_out[i, 2] = p[2]
        t_prev = t[i]


def _safe_gradient(z: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Like np.gradient(z, t, edge_order=2) but works for n<3."""
    n = z.shape[0]
    g = np.empty_like(z, dtype=np.float64)
    if n == 0:
        return g  # unused
    if n == 1:
        g[0] = 0.0
        return g
    if n == 2:
        dt = t[1] - t[0] if t[1] != t[0] else 1.0
        val = (z[1] - z[0]) / dt
        g[0] = val
        g[1] = val
        return g
    # n >= 3
    return np.gradient(z, t, edge_order=2)


def _count_bounces_from_traj(P_fit: np.ndarray, t: np.ndarray, table_z: float) -> int:
    """Counts downward crossings of z==table_z using a safe gradient."""
    n = len(t)
    if n < 2:
        return 0
    z = P_fit[:, 2]
    vz = _safe_gradient(z, t)
    count = 0
    for i in range(1, n):
        if (z[i - 1] > table_z) and (z[i] <= table_z) and (vz[i - 1] < 0.0):
            count += 1
    return count


def simulate_with_bounces_fast(p0, v0, w0, t, table_z=TABLE_Z, max_bounces=2, substep=0.002):
    """Wrapper that dispatches to JIT if available."""
    t = np.asarray(t, dtype=np.float64)
    P = np.empty((len(t), 3), dtype=np.float64)
    if NUMBA_OK:
        simulate_with_bounces_jit(
            np.asarray(p0, dtype=np.float64),
            np.asarray(v0, dtype=np.float64),
            np.asarray(w0, dtype=np.float64),
            t.astype(np.float64),
            float(table_z),
            int(max_bounces),
            float(substep),
            P,
        )
    else:
        p = np.array(p0, dtype=float)
        v = np.array(v0, dtype=float)
        w = np.array(w0, dtype=float)
        P[0] = p
        t_prev = t[0]
        bounces = 0

        def accel(v_, w_):
            speed = np.linalg.norm(v_)
            return G + (-KD * speed * v_ + KM * np.cross(w_, v_)) / M

        def rk4(p_, v_, w_, dt_):
            a1 = accel(v_, w_)
            k1p, k1v = v_, a1
            v2 = v_ + 0.5 * dt_ * k1v
            p_ + 0.5 * dt_ * k1p
            a2 = accel(v2, w_)
            k2p, k2v = v2, a2
            v3 = v_ + 0.5 * dt_ * k2v
            p_ + 0.5 * dt_ * k2p
            a3 = accel(v3, w_)
            k3p, k3v = v3, a3
            v4 = v_ + dt_ * k3v
            p_ + dt_ * k3p
            a4 = accel(v4, w_)
            k4p, k4v = v4, a4
            p_n = p_ + (dt_ / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
            v_n = v_ + (dt_ / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
            return p_n, v_n

        def bounce(v_, w_):
            sv = np.hypot(v_[0] - R * w_[1], v_[1] + R * w_[0])
            a = 0.0 if sv < EPS else MU * (1.0 + COR) * abs(v_[2]) / sv
            if a > ALPHA_ROLL_THRESHOLD:
                A = np.diag([0.6, 0.6, -COR])
                B = np.array([[0, 0.4 * R, 0], [-0.4 * R, 0, 0], [0, 0, 0]])
                C = np.array([[0, -0.6 / R, 0], [0.6 / R, 0, 0], [0, 0, 0]])
                D = np.diag([0.4, 0.4, 1.0])
            else:
                A = np.diag([1 - a, 1 - a, -COR])
                B = np.array([[0, a * R, 0], [-a * R, 0, 0], [0, 0, 0]])
                C = np.array([[0, -1.5 * a / R, 0], [1.5 * a / R, 0, 0], [0, 0, 0]])
                D = np.diag([1 - 1.5 * a, 1 - 1.5 * a, 1.0])
            return A @ v_ + B @ w_, C @ v_ + D @ w_

        for i in range(1, len(t)):
            dt_total = t[i] - t_prev
            if dt_total == 0.0:
                P[i] = p
                continue
            steps = max(1, int(np.ceil(abs(dt_total) / substep)))
            h = dt_total / steps
            for _ in range(steps):
                p_old, v_old = p, v
                p, v = rk4(p, v, w, h)
                if bounces < max_bounces and (p_old[2] > table_z) and (p[2] <= table_z) and (v_old[2] < 0.0):
                    denom = p[2] - p_old[2]
                    s = 0.0 if abs(denom) < 1e-12 else np.clip((table_z - p_old[2]) / denom, 0.0, 1.0)
                    p_hit, v_hit = rk4(p_old, v_old, w, h * s)
                    p_hit[2] = table_z
                    v, w = bounce(v_hit, w)
                    p, v = rk4(p_hit, v, w, h * (1.0 - s))
                    bounces += 1
            P[i] = p
            t_prev = t[i]
    return P


@dataclass
class FitResult:
    p0: np.ndarray
    v0: np.ndarray
    w0: np.ndarray
    speed: float
    spin: float
    traj_pred: np.ndarray
    cost: float
    success: bool
    message: str
    n_bounces_pred: int


def fit_initial_state(
    t: np.ndarray,
    P_obs: np.ndarray,
    p0_init: np.ndarray = None,
    v0_init: np.ndarray = None,
    w0_init: np.ndarray = None,
    bounds_pos: tuple[float, float] = (-10.0, 10.0),
    bounds_speed: tuple[float, float] = (0.0, 50.0),
    bounds_spin: tuple[float, float] = (0.0, 500.0),
    table_z: float = TABLE_Z,
    max_bounces: int = 2,
    substep: float = 0.002,
    robust_huber: bool = True,
    ignore_first_obs: bool = False,
    max_nfev: int | None = None,
):
    t = np.asarray(t, dtype=np.float64)
    P_obs = np.asarray(P_obs, dtype=np.float64)
    assert t.ndim == 1 and P_obs.shape == (len(t), 3)

    #  remove first observation from fitting if requested
    valid_idx = np.arange(len(P_obs))
    if ignore_first_obs:
        valid_idx = valid_idx[1:]

    # init defaults (still from the first non-NaN obs)
    if p0_init is None:
        p0_init = np.where(np.isnan(P_obs[0]), 0.0, P_obs[0])

    if v0_init is None:
        if len(t) >= 2:
            dt = max(t[1] - t[0], 1e-3)
            p1 = np.where(np.isnan(P_obs[1]), P_obs[0], P_obs[1])
            v0_init = (p1 - p0_init) / dt
        else:
            v0_init = np.zeros(3)

    if w0_init is None:
        w0_init = np.zeros(3)

    x0 = np.hstack([p0_init, v0_init, w0_init])

    # bounds
    p_lo, p_hi = bounds_pos
    v_cap = bounds_speed[1]
    w_cap = bounds_spin[1]
    lb = np.array([p_lo, p_lo, p_lo, -v_cap, -v_cap, -v_cap, -w_cap, -w_cap, -w_cap], dtype=np.float64)
    ub = np.array([p_hi, p_hi, p_hi, v_cap, v_cap, v_cap, w_cap, w_cap, w_cap], dtype=np.float64)

    # residuals
    def residuals(x: np.ndarray) -> np.ndarray:
        p0 = x[0:3]
        v0 = x[3:6]
        w0 = x[6:9]
        P_pred = simulate_with_bounces_fast(p0, v0, w0, t, table_z=table_z, max_bounces=max_bounces, substep=substep)
        r = (P_pred[valid_idx] - P_obs[valid_idx]).ravel()
        return r

    # lsq_kwargs = dict(bounds=(lb, ub), xtol=1e-10, ftol=1e-10, gtol=1e-10, verbose=0)
    lsq_kwargs = dict(bounds=(lb, ub), xtol=1e-3, ftol=1e-3, gtol=1e-3, verbose=0)
    if robust_huber:
        lsq_kwargs.update(loss="huber", f_scale=0.1)
    if max_nfev is not None:
        lsq_kwargs["max_nfev"] = max_nfev

    sol = least_squares(residuals, x0, **lsq_kwargs)

    # final solution
    p0_hat, v0_hat, w0_hat = sol.x[0:3], sol.x[3:6], sol.x[6:9]
    P_fit = simulate_with_bounces_fast(
        p0_hat, v0_hat, w0_hat, t, table_z=table_z, max_bounces=max_bounces, substep=substep
    )

    # cost over fitted points only
    r = (P_fit[valid_idx] - P_obs[valid_idx]).ravel()
    # print(valid_idx)
    # print(P_fit)
    # print(P_obs)
    # print(P_fit[valid_idx] - P_obs[valid_idx])
    cost = 0.5 * float(np.dot(r, r))

    # count bounces safely
    n_b = _count_bounces_from_traj(P_fit, t, table_z)

    return FitResult(
        p0=p0_hat,
        v0=v0_hat,
        w0=w0_hat,
        speed=float(np.linalg.norm(v0_hat)),
        spin=float(np.linalg.norm(w0_hat)),
        traj_pred=P_fit,
        cost=cost,
        success=bool(sol.success),
        message=f"{sol.message} | ignore_first_obs={ignore_first_obs}",
        n_bounces_pred=int(n_b),
    )


def plot_xyz_trajectory(t, P_obs, P_pred, title="Ball trajectory fit"):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["X [m]", "Y [m]", "Z [m]"]
    for i, ax in enumerate(axes):
        mask = ~np.isnan(P_obs[:, i])
        ax.plot(t[mask], P_obs[mask, i], "o", markersize=4, alpha=0.6, label="observed")
        ax.plot(t, P_pred[:, i], "-", linewidth=2, label="predicted")
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title(title)
    axes[-1].set_xlabel("Time [s]")
    axes[0].legend()
    plt.tight_layout()
    plt.show()


def replace_zeros_with_nan(P: np.ndarray) -> np.ndarray:
    """Treat [0,0,0] as missing: replace with NaN (row-wise)."""
    P = np.asarray(P, dtype=np.float64).copy()
    zero_mask = (P[:, 0] == 0.0) & (P[:, 1] == 0.0) & (P[:, 2] == 0.0)
    P[zero_mask] = np.nan
    return P


def plot_obs_and_interpolated(
    t_obs: np.ndarray,
    P_obs: np.ndarray,
    t_query: np.ndarray,
    P_query: np.ndarray,
    title: str = "Observed vs. interpolated trajectory",
):
    t_obs = np.asarray(t_obs, dtype=np.float64)
    P_obs = np.asarray(P_obs, dtype=np.float64)
    t_query = np.asarray(t_query, dtype=np.float64)
    P_query = np.asarray(P_query, dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["X [m]", "Y [m]", "Z [m]"]

    for i, ax in enumerate(axes):
        # observations (mask NaNs)
        mask = ~np.isnan(P_obs[:, i])
        if mask.any():
            ax.plot(t_obs[mask], P_obs[mask, i], "o", markersize=4, alpha=0.7, label="observed")
        # interpolated predictions
        ax.plot(t_query, P_query[:, i], "-", linewidth=2, label="predicted (interp)")
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title(title)
            ax.legend()

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()


def filter_obs_and_time(segment: np.ndarray, fps: float):
    seg = np.asarray(segment, dtype=np.float64)
    n = seg.shape[0]
    t_full = np.arange(n, dtype=np.float64) / fps
    keep = ~((seg[:, 0] == 0.0) & (seg[:, 1] == 0.0) & (seg[:, 2] == 0.0))
    return t_full[keep], seg[keep]


if __name__ == "__main__":
    FPS = 30.0  # adjust if needed
    ignore_first = True

    with open("ball_segments.pkl", "rb") as f:
        segments = pickle.load(f)

    for seg_idx, segment in enumerate(segments[4:], start=4):
        # 1) Filter P and t with SAME mask (preserves original timing)
        t_orignal = np.arange(len(segment)) / FPS
        t, P_obs = filter_obs_and_time(segment, FPS)

        if P_obs.shape[0] < 2:
            print(f"[seg {seg_idx}] Skipping: fewer than 2 valid observations after filtering.")
            continue

        # 2) Initialize from the first two kept samples (their original dt)
        dt01 = max(t[1] - t[0], 1e-3)
        if ignore_first:
            p0_init = P_obs[1]
            v0_init = (P_obs[2] - P_obs[1]) / dt01
        else:
            p0_init = P_obs[0]
            v0_init = (P_obs[1] - P_obs[0]) / dt01
        w0_init = np.zeros(3, dtype=np.float64)

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
            ignore_first_obs=ignore_first,
        )
        # res, inliers = fit_initial_state_ransac(
        #     t, P_obs,
        #     table_z=TABLE_Z,
        #     max_bounces=2,
        #     substep=0.002,
        #     min_samples=4,
        #     residual_threshold=0.10,   # 2 cm per-frame 3D error
        #     max_trials=100,
        #     stop_inliers_ratio=0.9,
        #     random_state=0,
        # )

        print(f"[seg {seg_idx}] Success: {res.success} | {res.message}")
        print("  Estimated p0:", res.p0)
        print("  Estimated v0:", res.v0, "| speed =", res.speed, "m/s")
        print("  Estimated w0:", res.w0, "| spin  =", res.spin, "rad/s")

        # 4) Metrics & plot on filtered data
        rmse = float(np.sqrt(np.mean((res.traj_pred - P_obs) ** 2)))
        print("  Final RMSE [m]:", rmse)

        interpolated = simulate_with_bounces_fast(
            res.p0,
            res.v0,
            res.w0,
            t_orignal,
            table_z=TABLE_Z,
        )
        # plot_xyz_trajectory(t, P_obs, res.traj_pred, title=f"Segment {seg_idx}")
        plot_obs_and_interpolated(t, P_obs, t_orignal, interpolated)
