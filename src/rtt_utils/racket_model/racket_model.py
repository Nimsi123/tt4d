import casadi as ca

from .constants import I, M, R


def cas_elastic_bounce(v_in, w_in, cor=0.8, friction_coef=0.0018):
    """CasADi-compatible elastic bounce model"""
    # Compute coefficients
    kp = friction_coef
    kv = kp / M
    kw = kp / I
    print("kw", kw)
    print("kv", kv)

    # Define matrices symbolically
    Av = ca.vertcat(
        ca.horzcat(1 - kv, 0, 0),
        ca.horzcat(0, 1 - kv, 0),
        ca.horzcat(0, 0, -cor),
    )
    Bv = ca.vertcat(
        ca.horzcat(0, kv * R, 0),
        ca.horzcat(-kv * R, 0, 0),
        ca.horzcat(0, 0, 0),
    )

    Aw = ca.vertcat(
        ca.horzcat(0, -kw * R, 0),
        ca.horzcat(kw * R, 0, 0),
        ca.horzcat(0, 0, 0),
    )

    Bw = ca.vertcat(
        ca.horzcat(1 - kw * R**2, 0, 0),
        ca.horzcat(0, 1 - kw * R**2, 0),
        ca.horzcat(0, 0, 1),
    )

    # Apply the model
    v_out = ca.mtimes(Av, v_in) + ca.mtimes(Bv, w_in)
    w_out = ca.mtimes(Aw, v_in) + ca.mtimes(Bw, w_in)

    return v_out, w_out
