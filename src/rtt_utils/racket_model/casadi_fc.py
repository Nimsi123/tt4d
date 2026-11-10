"""
Casadi implementation of the dynamics
The dynamics are modeled as a Differential Algebraic Equation (DAE)
"""

import casadi as ca

from .constants import COR, KD, KM, MU, M, R

G = ca.DM([0, 0, -9.81])  # Gravity acceleration (m/s^2)


b = ca.DM([0, 0, -9.81])  # gravitational acceleration


def quat_to_rotmat(q):
    """Convert CasADi quaternion [x, y, z, w] to rotation matrix."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.horzcat(1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        ca.horzcat(2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)),
        ca.horzcat(2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)),
    )


def rk4_ball_step(pos0, vel0, spin0, dt):
    k1_v, k1_a = ball_dynamics(vel0, spin0)
    k2_v, k2_a = ball_dynamics(vel0 + 0.5 * dt * k1_v, spin0)
    k3_v, k3_a = ball_dynamics(vel0 + 0.5 * dt * k2_v, spin0)
    k4_v, k4_a = ball_dynamics(vel0 + dt * k3_v, spin0)

    vel = vel0 + (dt / 6.0) * (k1_a + 2 * k2_a + 2 * k3_a + k4_a)
    pos = pos0 + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    return ca.vertcat(pos, vel)


def rk4_ball_traj(pos0, vel0, spin0, dt, N):
    pos = pos0
    vel = vel0
    positions = [pos]
    velocities = [vel]
    for _ in range(N):
        k1_v, k1_a = ball_dynamics(vel, spin0)
        k2_v, k2_a = ball_dynamics(vel + 0.5 * dt * k1_v, spin0)
        k3_v, k3_a = ball_dynamics(vel + 0.5 * dt * k2_v, spin0)
        k4_v, k4_a = ball_dynamics(vel + dt * k3_v, spin0)

        vel = vel + (dt / 6.0) * (k1_a + 2 * k2_a + 2 * k3_a + k4_a)
        pos = pos + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        positions.append(pos)
        velocities.append(vel)
    return ca.horzcat(*positions), ca.horzcat(*velocities)


def euler_ball_traj(pos0, vel0, spin0, dt, N):
    pos = pos0
    vel = vel0
    positions = [pos]
    velocities = [vel]
    for _ in range(N):
        acc = ball_dynamics(vel, spin0)[1]  # acceleration only
        vel = vel + dt * acc
        pos = pos + dt * vel
        positions.append(pos)
        velocities.append(vel)
    return ca.horzcat(*positions), ca.horzcat(*velocities)


def expm_approx(A, order=10):
    # Taylor series approximation of exp(A)
    result = ca.DM.eye(3)
    term = ca.DM.eye(3)
    for i in range(1, order):
        term = ca.mtimes(term, A) / i
        result += term
    return result


def cas_cross_mat(w):
    """Skew-symmetric matrix for cross product"""
    return ca.vertcat(
        ca.horzcat(0, -w[2], w[1]),
        ca.horzcat(w[2], 0, -w[0]),
        ca.horzcat(-w[1], w[0], 0),
    )


def cas_get_A(v0, w0):
    norm_v = ca.norm_2(v0)
    A_drag = -KD * norm_v / M * ca.DM.eye(3)
    A_magnus = KM / M * cas_cross_mat(w0)
    return A_drag + A_magnus


def cas_get_v(t, v0, w0):
    A = cas_get_A(v0, w0)
    A_inv = ca.inv(A)
    v_star = -A_inv @ b
    exp_At = expm_approx(A * t)
    return v_star + ca.mtimes(exp_At, v0 - v_star)


def cas_get_p(t, p0, v0, w0):
    A = cas_get_A(v0, w0)
    A_inv = ca.inv(A)
    v_star = -A_inv @ b
    # exp_At = ca.expm(A * t)
    exp_At = expm_approx(A * t)
    return ca.mtimes(A_inv, exp_At - ca.MX.eye(3)) @ (v0 - v_star) + v_star * t + p0


def build_cas_get_p_fun():
    t = ca.MX.sym("t")
    p0 = ca.MX.sym("p0", 3)
    v0 = ca.MX.sym("v0", 3)
    w0 = ca.MX.sym("w0", 3)

    A = cas_get_A(v0, w0)
    A_inv = ca.inv(A)
    v_star = -A_inv @ b
    exp_At = expm_approx(A * t)

    displacement = ca.mtimes((exp_At - ca.MX.eye(3)), v0 - v_star)
    p_t = ca.mtimes(A_inv, displacement) + v_star * t + p0

    return ca.Function("cas_get_p", [t, p0, v0, w0], [p_t])


def build_cas_get_v_fun():
    t = ca.MX.sym("t")
    v0 = ca.MX.sym("v0", 3)
    w0 = ca.MX.sym("w0", 3)

    A = cas_get_A(v0, w0)
    A_inv = ca.inv(A)
    v_star = -A_inv @ b
    exp_At = expm_approx(A * t)
    v_t = v_star + ca.mtimes(exp_At, v0 - v_star)

    return ca.Function("cas_get_v", [t, v0, w0], [v_t])


def ball_dynamics(v, w):
    """Compute ball dynamics with drag and Magnus effect."""
    eps = 1e-6  # To avoid calculating the jacobian of the sqrt at 0 (undefined)
    speed = ca.sqrt(ca.mtimes(v.T, v) + eps)
    drag_force = -KD * speed * v
    magnus_force = KM * ca.cross(w, v)
    a = G + drag_force / M + magnus_force / M
    return v, a


def get_surface_v(v, w):
    """Calculate the surface velocity for symbolic vectors v and w."""
    # CasADi compatible surface velocity calculation
    return ca.sqrt((v[0] - R * w[1]) ** 2 + (v[1] + R * w[0]) ** 2 + 1e-3)


def get_alpha(v, w):
    """Calculate the alpha value for symbolic vectors v and w."""
    # Compute the surface velocity (use CasADi expressions)
    surf_v = get_surface_v(v, w)

    # CasADi-compatible alpha calculation
    alpha = MU * (1 + COR) * ca.fabs(v[2]) / surf_v
    return alpha


def cas_bounce_model(v, w, smooth=False):
    """CasADi-compatible bounce dynamics for symbolic vectors v and w."""
    # assert v.shape == (3, 1) and w.shape == (3, 1)

    alpha = get_alpha(v, w)

    # Rolling matrices
    A_roll = ca.DM([[0.6, 0, 0], [0, 0.6, 0], [0, 0, -COR]])
    B_roll = ca.DM([[0, 0.4 * R, 0], [-0.4 * R, 0, 0], [0, 0, 0]])
    C_roll = ca.DM([[0, -0.6 / R, 0], [0.6 / R, 0, 0], [0, 0, 0]])
    D_roll = ca.DM([[0.4, 0, 0], [0, 0.4, 0], [0, 0, 1]])

    # Sliding matrices
    A_slide = ca.vertcat(
        ca.horzcat(1 - alpha, 0, 0),
        ca.horzcat(0, 1 - alpha, 0),
        ca.horzcat(0, 0, -COR),
    )
    B_slide = ca.vertcat(
        ca.horzcat(0, alpha * R, 0),
        ca.horzcat(-alpha * R, 0, 0),
        ca.horzcat(0, 0, 0),
    )
    C_slide = ca.vertcat(
        ca.horzcat(0, -3 / 2 * alpha / R, 0),
        ca.horzcat(3 / 2 * alpha / R, 0, 0),
        ca.horzcat(0, 0, 0),
    )
    D_slide = ca.vertcat(
        ca.horzcat(1 - 3 / 2 * alpha, 0, 0),
        ca.horzcat(0, 1 - 3 / 2 * alpha, 0),
        ca.horzcat(0, 0, 1),
    )

    if smooth:
        k = 200  # sharpness of transition; increase for a steeper change
        s = 1 / (1 + ca.exp(-k * (alpha - 0.4)))  # smooth switch ∈ (0,1)

        A = s * A_roll + (1 - s) * A_slide
        B = s * B_roll + (1 - s) * B_slide
        C = s * C_roll + (1 - s) * C_slide
        D = s * D_roll + (1 - s) * D_slide

    else:
        # Select rolling or sliding matrices
        A = ca.if_else(alpha > 0.4, A_roll, A_slide)
        B = ca.if_else(alpha > 0.4, B_roll, B_slide)
        C = ca.if_else(alpha > 0.4, C_roll, C_slide)
        D = ca.if_else(alpha > 0.4, D_roll, D_slide)

    # For debuging purposes: the ball bonces normally
    # A = ca.DM([[0, 0, 0], [0, 0, 0], [0, 0, -COR]])
    # B = ca.DM([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    # C = ca.DM([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    # D = ca.DM([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    # Compute va and wa using matrix multiplication
    va = ca.mtimes(A, v) + ca.mtimes(B, w)
    wa = ca.mtimes(C, v) + ca.mtimes(D, w)

    return va, wa
