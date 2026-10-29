#!/usr/bin/env python3
"""
qgl_pancake.py -- the non-singular "pancake" Big Bang of Dechant, Lasenby & Hobson
(arXiv:0809.4335): a massive scalar field in a biaxial Bianchi IX universe.

Metric (their normalisation, kappa = 8 pi G):

    ds^2 = -dt^2 + (1/4) sum_i R_i^2 (omega^i)^2 ,     R_2 = R_3

so R_i = 2 a_i in the convention of qgl_bkl (a_i multiplies sigma^i).  The matter is a scalar field phi with
V = m^2 phi^2 / 2 and a cosmological constant Lambda.  With H_i = R_i'/R_i, S = sum H_i and the slice
Ricci components R3_ii = 2 (R_i^4 - (R_j^2 - R_k^2)^2) / (R_1 R_2 R_3)^2 (the same as qgl_bkl.ricci3 with a = R/2):

    R_i'' = R_i ( -H_i (H_j + H_k) - R3_ii + kappa V + Lambda )
    phi'' = -S phi' - m^2 phi
    constraint:  sum_{i<j} H_i H_j + (1/2) sum_i R3_ii - kappa (phi'^2/2 + V) - Lambda = 0

These were derived independently of the paper (diagonal Bianchi IX, Einstein + scalar) and are checked against
the paper's equations (25)-(28) and its curvature at t = 0 (eq. 31) in tests/test_pancake.py.

The pancake solution is the odd-parity series
    R_1 = t (a0 + a2 t^2 + ...),  R_2 = R_3 = b0 + b2 t^2 + ...,  phi = f0 + f2 t^2 + ...
with a0, b0, f0 free.  R_1 -> 0 linearly while R_2 stays finite, so the volume vanishes at t = 0, but the
curvature invariants stay finite (K(0) is given in closed form by `kretschmann_at_pancake`); the shear diverges
as 1/t^2.  The solution continues through t = 0 with R_1 odd and R_2, phi even.

Extensions: `paper_parameters` (the paper's own run), `triaxial_run` (R_3 = R_2 (1 + delta) imposed at t_p > 0 and
followed both ways), and `x_geodesic` (geodesics winding round the closed x-direction).

`pancake_sensor_run` drives the QGL complexity scalar and Jaynes-Cummings sensor with K(t) along the comoving
worldline through the pancake.  That coupling is a modelling choice, as in qgl_curvature.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from unified_sandbox_core import Config
import qgl_bkl as bk
import qgl_curvature as qc


# --------------------------------------------------------------------------
#  Equations
# --------------------------------------------------------------------------

def slice_ricci(R) -> np.ndarray:
    """R3_ii for scale factors R_i (shape (..., 3)) in the paper's normalisation; equals bk.ricci3(R / 2).

    `qgl_bkl.ricci3` evaluates its numerator in factored form, (q_i - q_j + q_k)(q_i - q_k + q_j), differencing
    the large equal terms first.  The expanded form q_i^2 - (q_j - q_k)^2 loses all its digits near the pancake
    (R_1 -> 0 while R_2 = R_3 stay large), which first showed up as a 1e-4 constraint violation at b0 = 3 and a
    20% error in K at the paper's b0 = 18000.
    """
    return bk.ricci3(np.asarray(R, dtype=float) / 2.0)


def pancake_rhs(y, m: float, Lam: float = 0.0, kappa: float = 1.0) -> np.ndarray:
    """
    Time derivative of the state y = (R1, R2, R3, R1', R2', R3', phi, phi').  Regular at R1 = 0: the
    H_1 = R1'/R1 factor only multiplies R2', R3', which vanish there.
    """
    R, Rd, ph, phd = y[:3], y[3:6], y[6], y[7]
    V = 0.5 * m * m * ph * ph
    r3 = slice_ricci(R)
    H = Rd / R
    Rdd = np.empty(3)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        # R_i H_i = R_i' , so R_i'' = -R_i' (H_j + H_k) + R_i (...): only H_1 = R_1'/R_1 can be large, and it
        # multiplies R_2', R_3' ~ t
        Rdd[i] = -Rd[i] * (H[j] + H[k]) + R[i] * (-r3[i] + kappa * V + Lam)
    phdd = -np.sum(H) * phd - m * m * ph
    return np.concatenate([Rd, Rdd, [phd, phdd]])


def constraint(y, m: float, Lam: float = 0.0, kappa: float = 1.0) -> float:
    """Hamiltonian constraint (zero on solutions) for the state y."""
    R, Rd, ph, phd = y[:3], y[3:6], y[6], y[7]
    H = Rd / R
    V = 0.5 * m * m * ph * ph
    return float(H[0] * H[1] + H[1] * H[2] + H[0] * H[2] + 0.5 * np.sum(slice_ricci(R))
                 - kappa * (0.5 * phd * phd + V) - Lam)


# --------------------------------------------------------------------------
#  Series about the pancake
# --------------------------------------------------------------------------

def pancake_series(a0: float, b0: float, f0: float, m: float, Lam: float = 0.0, kappa: float = 1.0) -> Dict[str, float]:
    """
    Next coefficients of the odd-parity series (eq. 30 of the paper), solved from the field equations:

        b2 = (2 Lam b0^2 + kappa m^2 f0^2 b0^2 - 8) / (8 b0),   f2 = -m^2 f0 / 4,
        a2 = a0 (b0 (2 Lam + kappa m^2 f0^2) - 8 b2) / (12 b0)

    The constraint holds identically at t = 0 for these (checked in the tests).
    """
    b2 = (2.0 * Lam * b0 ** 2 + kappa * m * m * f0 ** 2 * b0 ** 2 - 8.0) / (8.0 * b0)
    f2 = -m * m * f0 / 4.0
    a2 = a0 * (b0 * (2.0 * Lam + kappa * m * m * f0 ** 2) - 8.0 * b2) / (12.0 * b0)
    return dict(a0=a0, a2=a2, b0=b0, b2=b2, f0=f0, f2=f2)


def series_state(c: Dict[str, float], t: float) -> np.ndarray:
    """State (R, R', phi, phi') from the series through O(t^2) in R_2, phi and O(t^3) in R_1."""
    R1 = t * (c["a0"] + c["a2"] * t * t)
    R2 = c["b0"] + c["b2"] * t * t
    return np.array([R1, R2, R2, c["a0"] + 3.0 * c["a2"] * t * t, 2.0 * c["b2"] * t, 2.0 * c["b2"] * t,
                     c["f0"] + c["f2"] * t * t, 2.0 * c["f2"] * t])


# --------------------------------------------------------------------------
#  Solution through the pancake
# --------------------------------------------------------------------------

def pancake_solution(a0: float, b0: float, f0: float, m: float, Lam: float = 0.0, kappa: float = 1.0,
                     t_max: float = 5.0, t_min: Optional[float] = None, t_start: float = 1e-4,
                     n: int = 400) -> Dict[str, np.ndarray]:
    """
    The pancake solution on t in [t_min, t_max] (t_min defaults to -t_max), skipping the interval
    |t| < t_start around the pancake itself, where the series supplies the starting data.  Both halves are
    integrated away from the pancake on a geometric grid (so the neighbourhood of t = 0 is resolved).

    Returns arrays on a time grid t (ascending, no t = 0 point):
        t, R (n, 3), Rd (n, 3), phi, phid, Rdd (n, 3)
    R_1 is negative for t < 0 (odd parity).
    """
    t_min = -t_max if t_min is None else t_min
    if not (t_min < -t_start and t_max > t_start):
        raise ValueError("need t_min < -t_start and t_max > t_start")
    c = pancake_series(a0, b0, f0, m, Lam, kappa)

    def half(sign: float, span: float) -> Dict[str, np.ndarray]:
        grid = np.geomspace(t_start, span, n)
        s0 = series_state(c, sign * t_start)
        # in s = sign * t the state (which holds t-derivatives) obeys dy/ds = sign * rhs(y)
        sol = solve_ivp(lambda s, y: sign * pancake_rhs(y, m, Lam, kappa),
                        [t_start, span], s0, t_eval=grid, method="DOP853", rtol=1e-11, atol=1e-13)
        if not sol.success:
            raise RuntimeError(f"pancake integration failed: {sol.message}")
        return dict(t=sign * sol.t, y=sol.y.T)

    fwd = half(+1.0, t_max)
    bwd = half(-1.0, -t_min)
    t = np.concatenate([bwd["t"][::-1], fwd["t"]])
    y = np.vstack([bwd["y"][::-1], fwd["y"]])
    Rdd = np.array([pancake_rhs(s, m, Lam, kappa)[3:6] for s in y])
    return dict(t=t, R=y[:, :3], Rd=y[:, 3:6], phi=y[:, 6], phid=y[:, 7], Rdd=Rdd,
                params=dict(a0=a0, b0=b0, f0=f0, m=m, Lam=Lam, kappa=kappa))


# --------------------------------------------------------------------------
#  Curvature
# --------------------------------------------------------------------------

def pancake_kretschmann(sol: Dict[str, np.ndarray]) -> np.ndarray:
    """Kretschmann scalar along a solution (geometry only, via qgl_bkl.bianchi9_kretschmann with a = |R|/2)."""
    R = sol["R"]
    sgn = np.sign(R)
    return bk.bianchi9_kretschmann(np.abs(R) / 2.0, sgn * sol["Rd"] / 2.0, sgn * sol["Rdd"] / 2.0)


def kretschmann_at_pancake(b0: float, f0: float, m: float, Lam: float = 0.0, kappa: float = 1.0,
                           a0: float = 1.0) -> float:
    """
    Closed-form Kretschmann scalar at t = 0, from the series (a2 = 2 a0 / (3 b0^2) always):

        e = 2 b2 / b0 = Lam/2 + kappa m^2 f0^2 / 4 - 2/b0^2
        K(0) = 16 e^2 + (128 - 48 a0^2) / b0^4

    Orthonormal-frame pieces: R_0202 = R_0303 and R_1212 = R_1313 equal e (each counted 4 times), R_0101 and
    R_2323 equal 4/b0^2, and the magnetic (Codazzi) part is a0 (-2, 1, 1)/b0^2, which enters with -8.  So K(0)
    depends on the pancake's opening speed a0 = R_1'(0), and is not sign-definite.  (An earlier version of this
    function left a0 out; that is exact only for a0 = 1 and was 20% off at the paper's a0 = 1.2.)
    The paper's eq. (31) lists only the *coordinate* components that survive at t = 0; the x-components vanish
    there only because g_xx -> 0, while their orthonormal-frame values are finite, so K needs them too.
    Checked against a brute-force Riemann tensor in tests/test_pancake.py.
    """
    e = 0.5 * Lam + 0.25 * kappa * m * m * f0 ** 2 - 2.0 / b0 ** 2
    return float(16.0 * e * e + (128.0 - 48.0 * a0 ** 2) / b0 ** 4)


def shear_squared(sol: Dict[str, np.ndarray]) -> np.ndarray:
    """sigma^2 = (1/2) sigma_ij sigma^ij = (1/6) sum_{i<j} (H_i - H_j)^2; -> 1/(3 t^2) at the pancake."""
    H = sol["Rd"] / sol["R"]
    return ((H[:, 0] - H[:, 1]) ** 2 + (H[:, 0] - H[:, 2]) ** 2 + (H[:, 1] - H[:, 2]) ** 2) / 6.0


# --------------------------------------------------------------------------
#  The paper's parameters and the inflation that follows the pancake
# --------------------------------------------------------------------------

def paper_parameters() -> Dict[str, float]:
    """
    The "realistic" parameters of Sec. V of arXiv:0809.4335 (kappa = 1): m = 1/64000, a0 = 1.2, b0 = 18000, f0 = 13.
    The page I read gives no value for Lambda, so it is set to 0 here.
    """
    return dict(a0=1.2, b0=18000.0, f0=13.0, m=1.0 / 64000.0, Lam=0.0, kappa=1.0)


def efolds(sol: Dict[str, np.ndarray], t_from: float = 0.0) -> float:
    """ln of the growth of the mean scale factor (R1 R2 R3)^(1/3) from the first sample at |t| >= t_from to the last."""
    vol = np.abs(np.prod(sol["R"], axis=1)) ** (1.0 / 3.0)
    i0 = int(np.argmax(sol["t"] >= t_from)) if t_from > 0 else int(np.argmin(np.abs(sol["t"])))
    return float(np.log(vol[-1] / vol[i0]))


# --------------------------------------------------------------------------
#  Triaxial perturbations
# --------------------------------------------------------------------------
#
# A perturbation R_3 = R_2 (1 + delta) switches on the term 8 delta / R_1^2 in the slice curvature
# (R3_22 ~ 4/b^2 - 8 delta/R_1^2, R3_33 ~ 4/b^2 + 8 delta/R_1^2 as R_1 -> 0), which is singular at the pancake itself.
# So the perturbation cannot be imposed *at* t = 0 with a regular series; it is imposed at t_p > 0 on the biaxial
# solution and followed forward (where the paper says the ratio freezes) and backward toward the pancake (where
# this module finds the curvature growing like delta^2 / t^4).  The constraint is restored by adjusting H_3 only.

def pancake_state_at(t: float, a0: float, b0: float, f0: float, m: float, Lam: float = 0.0, kappa: float = 1.0,
                     t_start: float = 1e-3) -> np.ndarray:
    """Biaxial pancake state (R, R', phi, phi') at t > 0, integrated from the series at t_start."""
    c = pancake_series(a0, b0, f0, m, Lam, kappa)
    s0 = series_state(c, t_start)
    sol = solve_ivp(lambda s, y: pancake_rhs(y, m, Lam, kappa), [t_start, t], s0, method="DOP853",
                    rtol=1e-12, atol=1e-14)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:, -1]


def perturb_triaxial(y, delta: float, m: float, Lam: float = 0.0, kappa: float = 1.0) -> np.ndarray:
    """
    R_3 -> R_3 (1 + delta), with H_1, H_2, phi, phi' unchanged and H_3 re-solved from the Hamiltonian constraint
    (linear in H_3): H_3 = (S - H_1 H_2) / (H_1 + H_2),  S = kappa (phi'^2 / 2 + V) + Lam - (1/2) sum R3_ii.
    """
    y = np.array(y, dtype=float)
    y[2] *= 1.0 + delta
    R = y[:3]
    H = y[3:6] / R
    V = 0.5 * m * m * y[6] ** 2
    S = kappa * (0.5 * y[7] ** 2 + V) + Lam - 0.5 * np.sum(slice_ricci(R))
    H3 = (S - H[0] * H[1]) / (H[0] + H[1])
    y[5] = H3 * R[2]
    return y


def _follow(y, t_from: float, t_to: float, m: float, Lam: float, kappa: float, n: int) -> Dict[str, np.ndarray]:
    grid = np.geomspace(t_from, t_to, n)
    if t_to < t_from:       # backward in t: integrate in s = -t, where the state's t-derivatives obey dy/ds = -rhs
        sol = solve_ivp(lambda s, yy: -pancake_rhs(yy, m, Lam, kappa), [-t_from, -t_to], y, t_eval=-grid,
                        method="DOP853", rtol=1e-11, atol=1e-14)
    else:
        sol = solve_ivp(lambda s, yy: pancake_rhs(yy, m, Lam, kappa), [t_from, t_to], y, t_eval=grid,
                        method="DOP853", rtol=1e-11, atol=1e-14)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    ys = sol.y.T
    Rdd = np.array([pancake_rhs(s, m, Lam, kappa)[3:6] for s in ys])
    return dict(t=grid, R=ys[:, :3], Rd=ys[:, 3:6], phi=ys[:, 6], phid=ys[:, 7], Rdd=Rdd)


def triaxial_run(a0: float, b0: float, f0: float, m: float, Lam: float = 0.0, kappa: float = 1.0,
                 delta: float = 1e-4, t_p: float = 1e3, t_forward: float = 3e6, t_back: float = 1e-1,
                 n: int = 200) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Perturb the pancake solution at t_p by `delta` and follow it forward to t_forward and back toward the pancake
    to t_back.  Returns {"forward": sol, "backward": sol} (each a dict as `pancake_solution`; t is in the order
    of integration, backward decreasing) plus "params".  Backward integration becomes ill-conditioned at very
    small t for large b0; t_p = 1e3, t_back = 0.1 works for the paper's parameters.
    """
    y = perturb_triaxial(pancake_state_at(t_p, a0, b0, f0, m, Lam, kappa), delta, m, Lam, kappa)
    out = dict(forward=_follow(y, t_p, t_forward, m, Lam, kappa, n),
               backward=_follow(y, t_p, t_back, m, Lam, kappa, n),
               params=dict(a0=a0, b0=b0, f0=f0, m=m, Lam=Lam, kappa=kappa, delta=delta, t_p=t_p))
    for k in ("forward", "backward"):
        out[k]["params"] = out["params"]
    return out


def anisotropy_ratio(sol: Dict[str, np.ndarray]) -> np.ndarray:
    """R_3 / R_2 - 1 along a solution."""
    return sol["R"][:, 2] / sol["R"][:, 1] - 1.0


# --------------------------------------------------------------------------
#  Geodesics winding around the closed x-direction
# --------------------------------------------------------------------------
#
# The metric does not depend on x, so p_x = (1/4) R_1^2 (x' + sin y z') = L is conserved.  A particle that starts
# with y' = z' = 0 keeps them (the frame equations preserve w_2 = w_3 = 0), and then
#
#     gamma = dt/dtau = sqrt(1 + 4 L^2 / R_1^2),   dx/dtau = 4 L / R_1^2,
#     dx/dt = 4 L / (|R_1| sqrt(R_1^2 + 4 L^2)) -> 2 sgn(L) / (a0 |t|)      as t -> 0,
#     dtau/dt = |R_1| / sqrt(R_1^2 + 4 L^2)       -> a0 |t| / (2 |L|).
#
# So x ~ -(2/a0) ln(-t) for t < 0 and (2/a0) ln t for t > 0: it diverges logarithmically at the pancake (the
# particle winds infinitely often round x), while the proper time spent within |t| < t1 is only a0 t1^2/(2|L|).
# This is the structure of eq. (46) of the paper (c3 = -c1 = 2/a0).

def x_geodesic(sol: Dict[str, np.ndarray], L: float) -> Dict[str, np.ndarray]:
    """
    Along a `pancake_solution` grid: dx/dt, dtau/dt, gamma, and x, tau accumulated from the first sample on
    each side of t = 0 (x and tau are zero at |t| = smallest |t| on that side and increase with t).
    Keys: t, gamma, dxdt, dtaudt, x_rel, tau_rel, dxdlnt (= t dx/dt).
    """
    from scipy.integrate import cumulative_trapezoid
    t, R1 = sol["t"], sol["R"][:, 0]
    gamma = np.sqrt(1.0 + 4.0 * L * L / R1 ** 2)
    dxdt = 4.0 * L / (R1 ** 2 * gamma)
    dtaudt = 1.0 / gamma
    x_rel, tau_rel = np.zeros_like(t), np.zeros_like(t)
    neg, pos = t < 0, t > 0
    for side in (neg, pos):
        idx = np.flatnonzero(side)
        ts = t[idx]
        x_rel[idx] = cumulative_trapezoid(dxdt[idx], ts, initial=0.0)
        tau_rel[idx] = cumulative_trapezoid(dtaudt[idx], ts, initial=0.0)
    return dict(t=t, gamma=gamma, dxdt=dxdt, dtaudt=dtaudt, x_rel=x_rel, tau_rel=tau_rel, dxdlnt=t * dxdt)


def crossing_proper_time(a0: float, L: float, t1: float) -> float:
    """Proper time spent in |t| < t1 by an x-geodesic with momentum L, to leading order: a0 t1^2 / (2 |L|)."""
    return float(a0 * t1 * t1 / (2.0 * abs(L)))


def x_winding(a0: float, t_outer: float, t_inner: float) -> float:
    """
    Total change in x between t = -t_outer and +t_outer, excluding |t| < t_inner, to leading order:
    2 (2 / a0) ln(t_outer / t_inner).  Diverges as t_inner -> 0.  Divide by the period of x (4 pi for the
    standard Euler-angle S^3; this depends on the identification, which I have not checked against the paper)
    to count turns.
    """
    return float(2.0 * (2.0 / a0) * np.log(t_outer / t_inner))


# --------------------------------------------------------------------------
#  Sensor through the pancake
# --------------------------------------------------------------------------

def pancake_sensor_run(sol: Dict[str, np.ndarray], t_start: float, t_stop: float, rate_scale: float = 1.0,
                       cfg: Optional[Config] = None, K_ref: Optional[float] = None,
                       K_trans: Optional[float] = None, v_ref: float = 0.5, n: int = 300) -> pd.DataFrame:
    """
    Sensor on the comoving worldline from t_start to t_stop (through the pancake if they straddle zero).
    The sensor clock runs at rate_scale sensor units per unit of cosmic time (comoving proper time = t).  K is
    the Kretschmann scalar interpolated on the solution; K_ref (rho_hat = 1) and K_trans (C_Q = 1/2) default to
    K(0), so C_Q = 1/2 at the pancake: a modelling choice.  Output as qgl_curvature.sensor_run_along plus t.
    """
    p = sol["params"]
    K0 = kretschmann_at_pancake(p["b0"], p["f0"], p["m"], p["Lam"], p["kappa"], p["a0"])
    K_ref = K0 if K_ref is None else K_ref
    K_trans = K0 if K_trans is None else K_trans
    t_full = sol["t"]
    K_full = pancake_kretschmann(sol)
    # tabulate K on a grid that is dense near t = 0; insert the exact value at t = 0 (finite, by the series)
    t_all = np.concatenate([t_full[t_full < 0], [0.0], t_full[t_full > 0]])
    K_all = np.concatenate([K_full[t_full < 0], [K0], K_full[t_full > 0]])
    lo, hi = sorted((t_start, t_stop))
    mask = (t_all >= lo) & (t_all <= hi)
    t_use, K_use = t_all[mask], K_all[mask]
    if len(t_use) > n:
        keep = np.flatnonzero(t_use == 0.0)                      # the exact pancake point always stays
        idx = np.unique(np.concatenate([np.linspace(0, len(t_use) - 1, n).astype(int), keep]))
        t_use, K_use = t_use[idx], K_use[idx]
    if t_stop < t_start:
        t_use, K_use = t_use[::-1], K_use[::-1]
    tau = np.abs(t_use - t_use[0]) * rate_scale
    df = qc.sensor_run_along(tau, K_use, np.zeros_like(tau), cfg=cfg, K_ref=K_ref, K_trans=K_trans, v_ref=v_ref)
    df.insert(1, "t", t_use)
    return df


def main() -> None:
    a0, b0, f0, m = 1.0, 3.0, 2.0, 0.5
    sol = pancake_solution(a0, b0, f0, m, t_max=6.0, n=300)
    K = pancake_kretschmann(sol)
    print(f"pancake: a0={a0}, b0={b0}, f0={f0}, m={m}; K(0) = {kretschmann_at_pancake(b0, f0, m, a0=a0):.6g}")
    i0 = np.argmin(np.abs(sol["t"]))
    for j in (0, len(K) // 4, i0 - 1, i0, len(K) - 1):
        t = sol["t"][j]
        print(f"t={t:+10.3e}  R1={sol['R'][j, 0]:+10.3e}  R2={sol['R'][j, 1]:9.4f}  phi={sol['phi'][j]:7.4f}  "
              f"K={K[j]:.6g}  t^2*shear^2={t * t * shear_squared(sol)[j]:.4f}")
    df = pancake_sensor_run(sol, -4.0, 4.0, rate_scale=4.0)
    print(df.iloc[::50][["t", "tau", "K", "C_Q", "omega_c", "n", "sigma_z"]].to_string(
        index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
