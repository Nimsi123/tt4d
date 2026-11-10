import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from .casadi_fc import (
    quat_to_rotmat,
    rk4_ball_traj,
)
from .racket_model import cas_elastic_bounce
from .utils import (
    draw_table_tennis_table,
    load_pose_from_yaml,
    set_axes_equal,
    to_homogeneous,
)


def solve_stroke(hit_pos, vb_in, wb_in, target_pos, target_spin, tf, use_T_flight=False):
    # param use_T_flight: use hardcoded T_flight -- False means not to use hardcoded

    # ----- CONFIG -----
    MAX_SPIN = 2 * np.pi * 100
    net_height = 0.1525
    # Weights of the cost function
    w_t = 0
    w_pos = 1e3
    w_spin = 0.005

    # ----- SETUP TARGET AND INITIALS -----
    hit_pos_sym = ca.SX.sym("hit_pos", 3)
    target_pos_sym = ca.SX.sym("target_pos", 3)
    v_in_sym = ca.SX.sym("vb_in", 3)
    w_in_sym = ca.SX.sym("wb_in", 3)
    target_spin_sym = ca.SX.sym("target_spin", 3)
    T_flight = ca.SX.sym("tf", 1)
    # Tn_ini = 0.5 * t_flight_init
    Tn_ini = 0.6 * tf
    N1, N2 = 10, 5

    # ----- OPTIMIZATION VARIABLES -----
    Tn = ca.SX.sym("Tn")  # Time of net cross
    # T_flight = ca.SX.sym("T_flight")  # Time of second bounce
    dt1 = Tn / N1
    dt2 = (T_flight - Tn) / N2

    v_racket = ca.SX.sym("v_racket", 3)
    q_racket = ca.SX.sym("q_racket", 4)

    # ----- RACKET-RELATIVE BOUNCE -----
    # change of coordinates to racket frame
    R_racket = quat_to_rotmat(q_racket)
    v_ball_rel = ca.mtimes(R_racket.T, v_in_sym - v_racket)  # relative velocity of the ball relative to the racket
    w_ball_rel = ca.mtimes(R_racket.T, w_in_sym)
    v_bounce_local, w_bounce_local = cas_elastic_bounce(v_ball_rel, w_ball_rel)
    v_bounce_world = ca.mtimes(R_racket, v_bounce_local) + v_racket
    w_bounce_world = ca.mtimes(R_racket, w_bounce_local)

    # ----- TRAJECTORY 1: OVER NET -----
    pos_traj1, vel_traj1 = rk4_ball_traj(hit_pos_sym, v_bounce_world, w_bounce_world, dt1, N1)
    # Euler integration scheme
    # pos_traj1, vel_traj1 = euler_ball_traj(
    #     hit_pos_sym, v_bounce_world, w_bounce_world, dt1, N1
    # )

    # ----- TRAJECTORY 2: TO TARGET -----
    pos_traj2, _ = rk4_ball_traj(pos_traj1[:, -1], vel_traj1[:, -1], v_bounce_world, dt2, N2)
    # Euler integration scheme
    # pos_traj2, _ = euler_ball_traj(
    #     pos_traj1[:, -1], vel_traj1[:, -1], v_bounce_world, dt2, N2
    # )
    pos_traj = ca.horzcat(pos_traj1, pos_traj2)

    final_pos = pos_traj[:, -1]
    final_spin = w_bounce_world

    # ----- CASADI TRAJ FUNCTION -----
    traj_fun = ca.Function(
        "traj",
        [v_racket, q_racket, Tn, hit_pos_sym, v_in_sym, w_in_sym, T_flight],
        [pos_traj, final_spin],
    )

    # ----- CONSTRAINTS -----
    g = [
        pos_traj1[1, -1],  # y at net
        pos_traj1[2, -1] - net_height,  # z > net
        ca.dot(q_racket, q_racket) - 1.0,  # quaternion unit norm
        ca.sumsqr(final_spin),  # spin^2 < max^2
        final_pos[2],  # z at second bounce ≈ 0
        R_racket[1, 2],
        R_racket[:, 2].T @ v_racket,
    ]

    g = ca.vertcat(*g)
    lbg = np.zeros(g.shape[0])
    ubg = np.zeros(g.shape[0])
    constraint_labels = {
        0: "net-cross y",
        1: "net-cross height > net",
        2: "quaternion unit norm",
        3: "spin ≤ 900 rad/s",
        4: "target position z ≈ 0",
        5: "Racket facing forward",
        6: "Racket and velocity same direction",
    }
    lbg[0] = -0.000
    ubg[0] = 0.000
    lbg[1] = 0.10
    ubg[1] = 0.5
    ubg[3] = MAX_SPIN**2
    lbg[4] = 0.0
    ubg[4] = 0.0
    lbg[5] = 0.0
    ubg[5] = ca.inf
    lbg[6] = 0.0
    ubg[6] = ca.inf

    # ----- COST FUNCTION -----
    cost = 0
    cost += w_spin * ca.sumsqr(final_spin - target_spin_sym)
    cost += w_pos * ca.sumsqr(final_pos - target_pos_sym)
    cost += w_t * T_flight

    # ----- SOLVER SETUP -----
    if use_T_flight:
        # T_flight as a fix value
        opt_vars = ca.vertcat(v_racket, q_racket, Tn)
        p = ca.vertcat(hit_pos_sym, target_pos_sym, v_in_sym, w_in_sym, target_spin_sym, T_flight)
    else:
        # T_flight as an optimization variable
        # need to make many changes afterward
        opt_vars = ca.vertcat(v_racket, q_racket, Tn, T_flight)
        p = ca.vertcat(hit_pos_sym, target_pos_sym, v_in_sym, w_in_sym, target_spin_sym)
    print(p.shape)
    nlp = {"x": opt_vars, "f": cost, "g": g, "p": p}

    ## For interior point solver
    solver = ca.nlpsol(
        "solver",
        "ipopt",
        nlp,
        {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 1000,
            "ipopt.tol": 1e-2,
            "ipopt.constr_viol_tol": 1e-3,
            "ipopt.dual_inf_tol": 1e-3,
            "ipopt.compl_inf_tol": 1e-3,
            # "ipopt.linear_solver": "paradiso"  # Fast, open-source, parallel
            # "ipopt.hessian_approximation": "limited-memory",
            # "ipopt.mu_strategy": "adaptive",
        },
    )

    ## For SQP solver
    # solver = ca.nlpsol(
    #     "solver",
    #     "sqpmethod",  # CSQP solver
    #     nlp,
    #     {
    #         "qpsol": "qpoases",  # You can also try 'osqp', 'hpipm', or 'qrqp', qpoases
    #         "qpsol_options.verbose": False,
    #         "qpsol_options.print_time": False,
    #         "qpsol_options.print_problem": False,
    #         "qpsol_options.print_in": False,
    #         "qpsol_options.print_out": False,
    #         # "qpsol_options.print_level": 1,
    #         "print_status": False,
    #         "print_header": False,
    #         "print_iteration": False,
    #         "print_time": False,
    #         "print_iteration": False,
    #         "print_in": False,
    #         "print_out": False,
    #         "verbose": False,
    #         # "max_iter": 20,
    #         # "tol_stat": 1e-2,
    #         # "tol_eq": 1e-3,
    #         # "tol_ineq": 1e-3,
    #         # "tol_comp": 1e-3,
    #     },
    # )
    # solver.print_options()

    if use_T_flight:
        lbx = [-5] * 3 + [-1] * 4 + [0] * 1  # State upper bound
        ubx = [5] * 3 + [1] * 4 + [2] * 1  # State lower bound
        x0 = np.array([0.0, 1.5, 0, -0.707, 0, 0, 0.707, Tn_ini])  # State init
        p_val = np.concatenate([hit_pos, target_pos, vb_in, wb_in, target_spin, [tf]]).reshape(-1, 1)
        print(p_val.shape)
    else:
        lbx = [-5] * 3 + [-1] * 4 + [0] * 2  # State upper bound
        ubx = [5] * 3 + [1] * 4 + [2] * 2  # State lower bound
        x0 = np.array([-0.89, 1.13, 1.49, -0.4634336, -0.15620112, -0.04083805, 0.87129947, Tn_ini, tf])  # State init
        p_val = np.concatenate([hit_pos, target_pos, vb_in, wb_in, target_spin]).reshape(-1, 1)
        print(p_val.shape)

    # ----- SOLVE -----
    # Use codegen to generate c code
    # solver.generate_dependencies("stroke_ocp_rk4_ipopt.c", {"verbose": False})
    ## Import of compiled code using codegen solver for faster inference
    # solver = ca.nlpsol("solver", "sqpmethod", "./stroke_ocp_rk4_osqp.so")
    sol = solver(
        x0=x0,
        p=p_val,
        lbg=lbg,
        ubg=ubg,
        lbx=lbx,
        ubx=ubx,
    )

    if use_T_flight:
        stats = solver.stats()
        opt_sol = sol["x"].full().flatten()
        v_opt, q_opt = opt_sol[:3], opt_sol[3:7]
        tn_opt = opt_sol[7]
        t_flight_opt = tf
    else:
        stats = solver.stats()
        opt_sol = sol["x"].full().flatten()
        v_opt, q_opt = opt_sol[:3], opt_sol[3:7]
        tn_opt, t_flight_opt = opt_sol[7:]

    # ----- CONSTRAINT CHECKING -----
    g_val = ca.Function("g_fun", [opt_vars, p], [g])(sol["x"], p_val).full().flatten()
    violations = []
    for i, val in enumerate(g_val):
        if val < lbg[i] - 1e-4:
            violations.append((i, f"below lb by {lbg[i] - val:.4g}"))
        elif val > ubg[i] + 1e-4:
            violations.append((i, f"above ub by {val - ubg[i]:.4g}"))

    print("✅ All constraints within bounds." if not violations else "⚠️ Violated constraints:")
    for idx, msg in violations:
        name = constraint_labels.get(idx, f"Constraint {idx}")
        print(f"  - {name}: {msg}")

    print("\n--- SOLVER RESULT ---")
    print("Success:", stats["success"])
    print("Status:", stats["return_status"])
    print("Final cost:", sol["f"])
    print("V racket opt:", v_opt)
    print("Q racket opt:", q_opt)
    print("Rotation matrix:", quat_to_rotmat(q_opt))

    print("tn:", tn_opt, "t_flight:", t_flight_opt)

    # ----- TRAJECTORY EVAL -----
    if use_T_flight:
        traj_result, spin_result = traj_fun(v_opt, q_opt, tn_opt, hit_pos, vb_in, wb_in, tf)
        traj_result = traj_result.full()
        spin_result = spin_result.full()
    else:
        traj_result, spin_result = traj_fun(v_opt, q_opt, tn_opt, hit_pos, vb_in, wb_in, t_flight_opt)
        traj_result = traj_result.full()
        spin_result = spin_result.full()

    print("Final spin:", spin_result.T)
    print("Final position:", traj_result[:, -1])
    print("Bounce error:", target_pos - traj_result[:, -1])

    # ----- PLOTTING -----
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*traj_result, label="Ball Trajectory")
    ax.scatter(*hit_pos, color="green", label="Hit", s=50)
    ax.scatter(*target_pos, color="red", label="Target", s=50)

    # Racket orientation
    # q_opt = np.array([-0.707, 0, 0, 0.707])
    R_eval = R.from_quat(q_opt).as_matrix()
    origin = hit_pos
    axis_len = 0.2
    for i, color, label in zip(range(3), ["r", "g", "b"], ["X", "Y", "Z"]):
        end = origin + R_eval[:, i] * axis_len
        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            color,
            label=f"Racket {label}-axis",
        )

    # Racket velocity
    arrow_dir = v_opt / np.linalg.norm(v_opt)
    ax.quiver(
        *origin,
        *arrow_dir,
        length=np.linalg.norm(v_opt) * 0.1,
        color="purple",
        linewidth=2,
        label="Racket Velocity",
    )

    ax.set(
        xlabel="X [m]",
        ylabel="Y [m]",
        zlabel="Z [m]",
        title="Ball Trajectory with Bounce",
    )
    ax.legend()
    ax.grid()
    plt.tight_layout()
    set_axes_equal(ax)
    draw_table_tennis_table(ax)
    plt.show()
    # ----- RETURN RESULTS -----
    return {
        "success": stats["success"],
        "status": stats["return_status"],
        "cost": sol["f"].full().item(),
        "v_racket": v_opt,
        "q_racket": q_opt,
        "tn": tn_opt,
        "t_flight": t_flight_opt,
        "traj": traj_result,
        "final_spin": spin_result,
    }


if __name__ == "__main__":
    """Questions:
    1. incoming spin in the x versus y direction ... doesn't this make a big difference?
    2. input and output of the racket bounce model are in the racket frame
    """

    # x is width, y is length, table height at 0
    # the quantities below are defined in the table frame (not in the racket frame)
    # conversion to the racket frame is done within the function

    # sanity checks.

    # # 1. ball moves down the middle of the table, large topspin, want to go back down the middle with no spin
    # # you should angle your racket down
    # hit = np.array([0, -2.74 / 2, 0.23])
    # v_ball = np.array([0, -2, -0.5])
    # w_ball = np.array([100, 0, 0]) # topspin
    # target_pos = np.array([0, 0.8, 0.0])
    # target_spin = np.array([0, 0.0, 0])

    # # 2. ball moves down the middle of the table, large backspin, want to go back down the middle with no spin
    # you should angle your racket up
    # hit = np.array([0, -2.74 / 2, 0.23])
    # v_ball = np.array([0, -2, -0.5])
    # w_ball = np.array([-100, 0, 0]) # backspin
    # target_pos = np.array([0, 0.8, 0.0])
    # target_spin = np.array([0, 0.0, 0])

    # # 3. ball moves down the middle of the table, no spin, want to go back down the middle with no spin
    # hit = np.array([0, -2.74 / 2, 0.23])
    # v_ball = np.array([0, -2, -0.5])
    # w_ball = np.array([0, 0, 0])
    # target_pos = np.array([0, 0.8, 0.0])
    # target_spin = np.array([0, 0.0, 0])

    # # 4. ball moves down the middle of the table, side spin, want to go back down the middle
    # hit = np.array([0, -2.74 / 2, 0.23])
    # v_ball = np.array([0, -2, -0.5])
    # w_ball = np.array([0, 100, 0]) # side spin
    # target_pos = np.array([0, 0.8, 0.0])
    # target_spin = np.array([0, 0.0, 0])

    # # 5. ball moves down the middle of the table, side spin, want to go back down the middle
    # hit = np.array([0, -2.74 / 2, 0.23])
    # v_ball = np.array([0, -2, -0.5])
    # w_ball = np.array([0, -100, 0]) # side spin
    # target_pos = np.array([0, 0.8, 0.0])
    # target_spin = np.array([0, 0.0, 0])

    # # 6. ball moves down the middle of the table, no spin, want to go back down the middle with backspin
    # # NOTICE THAT RELATIVE TO THE HITTER, GOING BACK WITH BACKSPIN IS POSITIVE X SPIN
    # hit = np.array([0, -2.74 / 2, 0.23])
    # v_ball = np.array([0, -2, -0.5])
    # w_ball = np.array([0, 0, 0])
    # target_pos = np.array([0, 0.8, 0.0])
    # target_spin = np.array([300, 0.0, 0])

    # 7. ball moves down the middle of the table, no spin, want to go back down the middle with topspin
    # NOTICE THAT RELATIVE TO THE HITTER, GOING BACK WITH TOPSPIN IS NEGATIVE X SPIN
    hit = np.array([0, -2.74 / 2, 0.23])
    v_ball = np.array([0, -2, -0.5])
    w_ball = np.array([0, 0, 0])
    target_pos = np.array([0, 0.8, 0.0])
    target_spin = np.array([-300, 0.0, 0])  # leave with topspin

    # hit = np.array([0.66, -2.74 / 2 - 0.1, 0.23])  # where we hit the ball
    # v_ball = np.array([0, -2, -0.5]) # incoming ball lin velocity
    # w_ball = np.array([0, 0, -300]) # incoming ball ang velocity
    # target = np.array([-0.3, 0.8, 0.0])  # Landing position target
    # spin = np.array([300.0, 0.0, 50.0])  # Outgoing spin target
    tf = 0.7  # time of flight
    result = solve_stroke(hit, v_ball, w_ball, target_pos, target_spin, tf)
    print("Success:", result["success"])
    print("Cost:", result["cost"])
    print("Final position:", result["traj"][:, -1])
    print("Final spin:", result["final_spin"].T)

    # Load the transforms
    T_rob = load_pose_from_yaml(
        "/home/gossard/Code/tt_ws/src/tt_tracking/tt_cameras/camera_info/calib_06_2025_0/robot_pose.yaml"
    )
    T_table = load_pose_from_yaml(
        "/home/gossard/Code/tt_ws/src/tt_tracking/tt_cameras/camera_info/calib_06_2025_0/table_pose.yaml"
    )

    # Transformation from table to robot frame
    T_rob = to_homogeneous(T_rob)
    T_table = to_homogeneous(T_table)
    T = np.linalg.inv(T_rob) @ T_table
    # Transform velocity
    v_racket_table = result["v_racket"]
    v_racket_robot = T[:3, :3] @ v_racket_table

    # Transform orientation
    q_racket_table = result["q_racket"]
    r_table = R.from_quat(q_racket_table)  # assumes [x, y, z, w]
    R_table = r_table.as_matrix()

    # Apply rotation to transform orientation to robot frame
    R_robot = T[:3, :3] @ R_table
    q_racket_robot = R.from_matrix(R_robot).as_quat()

    # Output
    print()
    print("Serve parameters in robot frame")
    print("Velocity [robot frame]:", v_racket_robot)
    print("Quaternion [robot frame]:", q_racket_robot)
