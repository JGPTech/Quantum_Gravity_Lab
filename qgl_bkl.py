#!/usr/bin/env python3
"""
qgl_bkl.py -- BKL / Mixmaster dynamics: Kasner epochs, bounces, and curvature.

Near a generic cosmological singularity the vacuum Bianchi IX ("Mixmaster")
universe

    ds^2 = -dt^2 + a1^2 sigma1^2 + a2^2 sigma2^2 + a3^2 sigma3^2 ,   d sigma^i = -1/2 eps_ijk sigma^j ^ sigma^k

is a sequence of Kasner epochs a_i ~ t^(p_i).  The spatial-curvature term of
Bianchi IX is negligible inside an epoch and grows when the one expanding axis
(the one with p < 0, as t -> 0) gets large; it then triggers a bounce to a new
Kasner epoch.  This module provides

  * the BKL map between epochs, in two equivalent forms
        bkl_transition(p)   per-axis Landau-Lifshitz rule (exponent vectors)
        bkl_u_map(u)        u -> u-1 (u >= 2) or 1/(u-1) (1 < u < 2, a new era)
  * the vacuum Bianchi IX equations in log time, integrated numerically, so the
    bounces are *produced by the dynamics* rather than imposed
  * the exact Kretschmann scalar of the Bianchi IX metric, continuous through
    the bounces (validated against a brute-force Riemann tensor in tests)
  * a comoving observer's Jaynes-Cummings sensor and complexity scalar riding
    through the whole sequence (qgl_curvature machinery).

Log-time form.  With s = ln t, x_i = ln a_i, v_i = dx_i/ds = t a_i'/a_i:

    dx_i/ds = v_i
    dv_i/ds = v_i (1 - sum_j v_j) - t^2 R3_ii ,    R3_ii = [a_i^4 - (a_j^2 - a_k^2)^2] / (2 a1^2 a2^2 a3^2)
    constraint:  2 (v1 v2 + v2 v3 + v3 v1) + t^2 R3 = 0 ,  R3 = (2 sum a_i^2 a_j^2 - sum a_i^4) / (2 a1^2 a2^2 a3^2)

Inside a Kasner epoch the curvature term vanishes, sum v = 1 and v_i *is* the
exponent p_i, so the epochs appear as plateaus of v and a bounce as a jump.

Comoving observers (x^i = const, lapse 1) are geodesics of this synchronous
metric, so a sensor on one ages tau = t0 - t on the way to the singularity.
Needs only numpy/scipy (and pandas via qgl_curvature); EinsteinPy is not used here.
"""

from __future__ import annotations

import copy
import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from unified_sandbox_core import Config
import qgl_curvature as qc
import qgl_kasner as qk


# --------------------------------------------------------------------------
#  The BKL map
# --------------------------------------------------------------------------

def bkl_transition(p: Sequence[float]) -> np.ndarray:
    """
    Landau-Lifshitz bounce rule, applied to the per-axis exponent vector p.

    With p_l < 0 the single negative exponent,

        p_l' = |p_l| / (1 + 2 p_l) ,      p_i' = (p_i + 2 p_l) / (1 + 2 p_l)   for the other two,

    which again satisfies both Kasner conditions.  The negative exponent moves
    to the axis that was the middle one, and the old negative axis becomes the
    middle one: the roles rotate every bounce.
    """
    p = np.asarray(p, dtype=float)
    neg = np.where(p < 0)[0]
    if len(neg) != 1:
        raise ValueError("a generic Kasner epoch has exactly one negative exponent")
    l = int(neg[0])
    out = (p + 2.0 * p[l]) / (1.0 + 2.0 * p[l])
    out[l] = abs(p[l]) / (1.0 + 2.0 * p[l])
    return out


def bkl_u_map(u: float) -> Tuple[float, bool]:
    """
    One bounce in the Lifshitz-Khalatnikov parameter: u -> u - 1 if u >= 2
    (same era), u -> 1/(u - 1) if 1 < u < 2 (a new era starts).  Returns
    (u_next, new_era).  u = 1 and u = 2 are the special cases u -> 0 -> 1/0
    that leave the generic regime; they are rejected.
    """
    if u <= 1.0:
        raise ValueError("u must be > 1")
    if u >= 2.0:
        return u - 1.0, False
    return 1.0 / (u - 1.0), True


def era_sequence(u0: float, n_epochs: int) -> List[dict]:
    """The first n_epochs Kasner epochs from u0: u, era number, and the sorted exponents."""
    out, u, era = [], float(u0), 0
    for _ in range(n_epochs):
        out.append(dict(u=u, era=era, exponents=tuple(float(x) for x in qk.kasner_exponents(u))))
        u, new_era = bkl_u_map(u)
        era += int(new_era)
    return out


# --------------------------------------------------------------------------
#  Exact curvature of Bianchi IX
# --------------------------------------------------------------------------

def ricci3(a: np.ndarray) -> np.ndarray:
    """Diagonal 3-Ricci components R3_ii (orthonormal frame) of the Bianchi IX slice; a has shape (..., 3)."""
    q = np.asarray(a, dtype=float) ** 2
    P = q[..., 0] * q[..., 1] * q[..., 2]
    out = np.empty_like(q)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        # q_i^2 - (q_j - q_k)^2 = (q_i - q_j + q_k)(q_i - q_k + q_j); differencing the large terms first avoids
        # catastrophic cancellation when one scale factor is tiny (e.g. near a pancake)
        out[..., i] = ((q[..., i] - q[..., j]) + q[..., k]) * ((q[..., i] - q[..., k]) + q[..., j]) / (2.0 * P)
    return out


def _codazzi(a: np.ndarray, ad: np.ndarray) -> np.ndarray:
    """
    R_0ijk for (i, j, k) cyclic, i = 0, 1, 2 (the magnetic Weyl components B_ii), from
    c_i = [(A_j - A_k - A_i)(H_i - H_k) - (A_j - A_k + A_i)(H_i - H_j)] / 2,
    H = a'/a and A_l = a_l / (a_j a_k).  Overall sign is a convention; the three
    components sum to zero.
    """
    H = ad / a
    A = np.stack([a[..., l] / (a[..., (l + 1) % 3] * a[..., (l + 2) % 3]) for l in range(3)], axis=-1)
    out = np.empty_like(H)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        out[..., i] = 0.5 * ((A[..., j] - A[..., k] - A[..., i]) * (H[..., i] - H[..., k])
                             - (A[..., j] - A[..., k] + A[..., i]) * (H[..., i] - H[..., j]))
    return out


def bianchi9_kretschmann(a, ad, add) -> np.ndarray:
    """
    Kretschmann scalar R_abcd R^abcd of -dt^2 + sum a_i^2 sigma_i^2 from the scale
    factors, their first and second time derivatives (arrays of shape (..., 3)).
    Valid off-shell (it does not use the field equations).

    Orthonormal-frame components: R_0i0i = -a_i''/a_i;  R_ijij = R3_ijij + H_i H_j
    with R3_ijij = (R3_ii + R3_jj - R3_kk)/2;  and the Codazzi components R_0ijk
    (see `_codazzi`).  The contraction counts R_0i0i 4 times, R_ijij 4 times and
    each R_0ijk 8 times (with a minus sign).

    K is *not* sign-definite: in vacuum K = 8 (E.E - B.B) with E, B the electric and
    magnetic parts of the Weyl tensor, and at a bounce the magnetic part briefly wins.
    """
    a, ad, add = (np.asarray(v, dtype=float) for v in (a, ad, add))
    H = ad / a
    Rc = ricci3(a)
    K = 4.0 * np.sum((add / a) ** 2, axis=-1)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        R3 = 0.5 * (Rc[..., i] + Rc[..., j] - Rc[..., k])
        K = K + 4.0 * (R3 + H[..., i] * H[..., j]) ** 2
    return K - 8.0 * np.sum(_codazzi(a, ad) ** 2, axis=-1)


def bianchi9_weyl_invariants(a, ad, add):
    """
    The two quadratic Weyl invariants of the vacuum Bianchi IX metric, as (K, K*):

        K  = C_abcd C^abcd = 8 (E.E - B.B)       (= the Kretschmann scalar in vacuum)
        K* = C_abcd *C^abcd = 16 E.B             (the Pontryagin density, up to sign)

    with E_ii = a_i''/a_i and B_ii the Codazzi components.  Use on solutions of the
    vacuum equations only (it takes R_0i0i = Weyl).  Both are validated against a
    brute-force Riemann tensor of the actual numerical solution (tests/test_bkl.py).
    """
    a, ad, add = (np.asarray(v, dtype=float) for v in (a, ad, add))
    return bianchi9_kretschmann(a, ad, add), 16.0 * np.sum((add / a) * _codazzi(a, ad), axis=-1)


# --------------------------------------------------------------------------
#  Mixmaster dynamics
# --------------------------------------------------------------------------
#
# Why not integrate in t?  With the curvature term present the singularity is not
# exactly at the t = 0 of the initial Kasner fit, and a log-t integration drifts past
# it.  The volume V = a1 a2 a3 always reaches zero at the singularity, so the equations
# are integrated in eta = ln V (decreasing) for the state (x_i = ln a_i, H_i = a_i'/a_i):
#
#     dx_i/deta = H_i / S ,   dH_i/deta = (-H_i S - R3_ii) / S ,   S = sum_j H_j
#
# (the log-time equations in the module docstring, with v_i = t H_i).  The time to the
# singularity t is recovered in a second pass: d ln t / d eta = 1 / (S t) = 1 / sum v
# is O(1), and in the final Kasner epoch t = 1/S, which gives the starting value at the
# *end* of the run and lets ln t be integrated back up with full relative precision at
# every scale (tracking T - T_s directly would underflow over many decades).  All "t"
# quantities below mean that.

def _t2_ricci3(x: np.ndarray, t: float) -> np.ndarray:
    """t^2 * R3_ii for a_i = e^{x_i}."""
    return t * t * ricci3(np.exp(x))


def constraint(x, v, t: float) -> float:
    """Hamiltonian constraint 2 sum_{i<j} v_i v_j + t^2 R3 (zero on solutions); v_i = t H_i."""
    q = np.exp(2.0 * np.asarray(x, dtype=float))
    P = q[0] * q[1] * q[2]
    R3 = (2.0 * (q[0] * q[1] + q[1] * q[2] + q[2] * q[0]) - np.sum(q ** 2)) / (2.0 * P)
    return float(2.0 * (v[0] * v[1] + v[1] * v[2] + v[2] * v[0]) + t * t * R3)


def _rhs_eta(eta, y):
    x, H = y[:3], y[3:6]
    S = H.sum()
    R3 = ricci3(np.exp(x))
    return np.concatenate([H / S, (-H * S - R3) / S])


def initial_state(u: float, t0: float, t_bounce: float) -> Tuple[float, np.ndarray]:
    """
    Start in a Kasner epoch with parameter u at time t0, with amplitudes
    a_i = alpha_i t^(p_i) chosen so that the first bounce occurs near t_bounce (< t0).

    Axis 0 carries the negative exponent.  The curvature term of axis i is
    (1/2) r_i t^(4 p_i), with r_n = alpha_n^2 / (alpha_m alpha_l)^2 for the negative
    axis and r_m = r_l = 1 / alpha_n^2 for the other two (taking alpha_m = alpha_l).
    The negative axis grows toward the singularity: r_n = 2 t_bounce^(-4 p_n) puts its
    term at 1 at t_bounce.  The other two decay toward it but are *large* at t0 unless
    alpha_n is big, so alpha_n is set to make them <= 1e-6 there; the first epoch is then
    a clean Kasner epoch as long as t_bounce << t0 (the negative axis term at t0 is
    (t_bounce/t0)^(4|p_n|)/2).  The exponents are moved by the smallest amount that
    makes the Hamiltonian constraint hold exactly at the start (a Newton projection).
    Returns (eta0 = ln V0, y0 = [x1, x2, x3, H1, H2, H3]); the nominal t0 is only used for the fit.
    """
    p = np.array(qk.kasner_exponents(float(u)), dtype=float)             # (neg, mid, large)
    r_n = 2.0 * t_bounce ** (-4.0 * p[0])
    alpha_n = np.sqrt(max(t0 ** (4.0 * p[1]), t0 ** (4.0 * p[2])) / (2.0 * 1e-6))
    alpha_ml = np.sqrt(alpha_n / np.sqrt(r_n))                           # alpha_m alpha_l = alpha_n / sqrt(r_n)
    alpha = np.array([alpha_n, alpha_ml, alpha_ml])
    x = np.log(alpha) + p * np.log(t0)
    v = p.copy()
    for _ in range(4):                                  # minimal (Newton) projection onto C = 0
        C = constraint(x, v, t0)
        grad = 2.0 * (v.sum() - v)                      # dC/dv_i
        v = v - C * grad / np.dot(grad, grad)
    return float(x.sum()), np.concatenate([x, v / t0])


def first_epoch_term(u: float, t0: float, t_bounce: float) -> float:
    """
    Size of the negative-axis curvature term, (t_bounce / t0)^(4 |p_1|) / 2, in the first Kasner epoch at its start
    (see `initial_state`).  It must be small for the first epoch to be a clean Kasner epoch.
    """
    p1 = abs(qk.kasner_exponents(float(u))[0])
    return 0.5 * (t_bounce / t0) ** (4.0 * p1)


def clean_t0(u: float, t_bounce: float = 1.0, tol: float = 1e-6) -> float:
    """
    Starting time t0 for which `first_epoch_term` equals tol.  p_1 = -u / (1 + u + u^2) -> 0 as u grows, so a fixed
    t0 (such as 1e6) is only clean for moderate u: at u = 8.5 the term is already 3e-3 and the first epoch is lost.
    """
    p1 = abs(qk.kasner_exponents(float(u))[0])
    return float(t_bounce * (0.5 / tol) ** (1.0 / (4.0 * p1)))


class Mixmaster:
    """
    A numerical vacuum Bianchi IX solution, integrated from a Kasner epoch toward the
    singularity until the volume has shrunk by the factor t0 / t_end.  After construction
    `t` means time to the singularity (see above); `t0` is the starting value, which differs
    from the nominal one by the (small) singularity-time offset.
    """

    def __init__(self, u: float, t0: float, t_bounce: float, t_end: float,
                 rtol: float = 1e-11, atol: float = 1e-13):
        if not (t_end < t_bounce < t0):
            raise ValueError("need t_end < t_bounce < t0")
        term = first_epoch_term(u, t0, t_bounce)
        if term > 1e-3:
            warnings.warn(f"first Kasner epoch is not clean (curvature term {term:.2g} at t0 for u = {u}); "
                          f"use t0 >= clean_t0(u, t_bounce) = {clean_t0(u, t_bounce):.3g}", RuntimeWarning,
                          stacklevel=2)
        eta0, y0 = initial_state(u, t0, t_bounce)
        eta_end = eta0 - np.log(t0 / t_end)
        sol = solve_ivp(_rhs_eta, [eta0, eta_end], y0, method="DOP853",
                        rtol=rtol, atol=atol, dense_output=True)
        if not sol.success:
            raise RuntimeError(f"Mixmaster integration failed: {sol.message}")
        self._sol, self.u = sol, float(u)

        # second pass: ln t from d ln t / d eta = 1 / (S t), started at the end where t = 1/S
        def dlnt(eta, w):
            S = self._sol.sol(eta)[3:6].sum()
            return [1.0 / (S * np.exp(w[0]))]
        S_end = sol.y[3:6, -1].sum()
        sol_t = solve_ivp(dlnt, [eta_end, eta0], [-np.log(S_end)], method="DOP853",
                          rtol=rtol, atol=1e-12, dense_output=True)
        if not sol_t.success:
            raise RuntimeError(f"time-to-singularity integration failed: {sol_t.message}")
        eta_nodes = sol.t[::-1]                          # increasing
        lnt_nodes = sol_t.sol(eta_nodes)[0]
        if np.any(np.diff(lnt_nodes) <= 0.0):
            raise RuntimeError("time to singularity is not monotonic along the solution")
        self.t0, self.t_end = float(np.exp(lnt_nodes[-1])), float(np.exp(lnt_nodes[0]))
        # eta as a smooth function of ln t, on the solver's own nodes
        self._eta_of_lnt = CubicSpline(lnt_nodes, eta_nodes)

    def _y(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return t, self._sol.sol(self._eta_of_lnt(np.log(t)))

    def state(self, t):
        """(x, v) at time(s) t to the singularity: x = ln a_i, v_i = t H_i, each of shape (n, 3)."""
        t, y = self._y(t)
        return y[:3].T, y[3:6].T * t[:, None]

    def scale_factors(self, t):
        """a, a', a'' (each (n, 3)) at times t, with a'' from the vacuum field equations."""
        t, y = self._y(t)
        a, H = np.exp(y[:3].T), y[3:6].T
        R3 = ricci3(a)
        Hj, Hk = np.roll(H, -1, axis=1), np.roll(H, -2, axis=1)
        return a, a * H, a * (-H * (Hj + Hk) - R3)

    def kretschmann(self, t):
        """Signed Kretschmann scalar K (see bianchi9_kretschmann)."""
        return bianchi9_kretschmann(*self.scale_factors(t))

    def weyl_invariants(self, t):
        """(K, K*) at times t."""
        return bianchi9_weyl_invariants(*self.scale_factors(t))

    def weyl_magnitude(self, t):
        """
        sqrt(K^2 + K*^2) = |C_abcd C^abcd - i C_abcd *C^abcd| = 8 |sum_i (E_i + i B_i)^2|: a sign-free,
        smooth curvature scale that equals K in a Kasner epoch (B = 0) and does not vanish
        at a bounce, where K itself changes sign.
        """
        K, Ks = self.weyl_invariants(t)
        return np.hypot(K, Ks)

    def constraint(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        x, v = self.state(t)
        return np.array([constraint(x[i], v[i], t[i]) for i in range(len(t))])

    def potential_fraction(self, t) -> np.ndarray:
        """Largest |t^2 R3_ii|: small (<< 1) inside a Kasner epoch, O(1) during a bounce."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        x, _ = self.state(t)
        return np.max(np.abs((t * t)[:, None] * ricci3(np.exp(x))), axis=1)

    def bounce_times(self, n_grid: int = 40000, threshold: float = 0.05) -> np.ndarray:
        """
        Times (descending) at which the curvature term peaks above `threshold`: the bounces.  The default 0.05 misses
        the weak bounces at large u (their peak is ~0.03 at u = 8.5); use the epoch threshold (~2e-3) there.
        """
        t = np.exp(np.linspace(np.log(self.t0), np.log(self.t_end), n_grid))
        W = self.potential_fraction(t)
        peak = (W[1:-1] > W[:-2]) & (W[1:-1] >= W[2:]) & (W[1:-1] > threshold)
        return t[1:-1][peak]

    def epochs(self, n_grid: int = 20000, threshold: float = 1e-3) -> List[dict]:
        """
        Kasner epochs: maximal runs on a log-t grid where the curvature term is below
        `threshold`.  Each has t_start > t_end, the median exponent vector v, its
        negative axis and the span in ln t.
        """
        t = np.exp(np.linspace(np.log(self.t0), np.log(self.t_end), n_grid))
        _, v = self.state(t)
        quiet = self.potential_fraction(t) < threshold
        out, start = [], None
        for i in range(n_grid + 1):
            ok = i < n_grid and quiet[i]
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                seg = slice(start, i)
                vm = np.median(v[seg], axis=0)
                out.append(dict(t_start=float(t[start]), t_end=float(t[i - 1]), v=vm,
                                neg_axis=int(np.argmin(vm)), dln_t=float(np.log(t[start] / t[i - 1]))))
                start = None
        return out


# --------------------------------------------------------------------------
#  Moving observers: timelike geodesics of the Mixmaster metric
# --------------------------------------------------------------------------

def geodesic_frame_rhs(w, a, H, branch: int = +1):
    """
    Timelike geodesic of -dt^2 + sum a_i^2 sigma_i^2 in the orthonormal frame e_i = xi_i / a_i.

    w_i are the frame components of the 4-velocity (u = gamma d_t + sum w_i e_i, gamma = sqrt(1 + w.w)),
    a the scale factors and H = a'/a at the observer's time.  The covariant momenta P_i = a_i w_i along the
    left-invariant fields xi_i obey an Euler-type equation (the group is SU(2)), and the expansion enters only
    through the change of frame, so

        dw_i/dtau = w_j w_k (a_j^2 - a_k^2) / (a_1 a_2 a_3)  -  (dt/dtau) H_i w_i ,   (i, j, k) cyclic

    with dt/dtau = branch * gamma: +1 for the expanding direction, -1 toward the singularity.  The first term
    is the only place the curvature of the slices acts; it vanishes when the scale factors are equal.
    Returns (dw/dtau, dt/dtau).  Validated against a coordinate-Christoffel integration in tests/test_bkl_geodesic.py.
    """
    w, a, H = (np.asarray(v, dtype=float) for v in (w, a, H))
    gamma = np.sqrt(1.0 + w @ w)
    V = a[0] * a[1] * a[2]
    dt = branch * gamma
    dw = np.empty(3)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        dw[i] = w[j] * w[k] * (a[j] ** 2 - a[k] ** 2) / V - dt * H[i] * w[i]
    return dw, dt


def mixmaster_geodesic(mm: "Mixmaster", w0: Sequence[float], t) -> dict:
    """
    Timelike geodesic of `mm` toward the singularity, started at t[0] with frame 4-velocity components w0
    (the velocity relative to the comoving observers, in units of c; w0 = 0 is the comoving observer) and
    sampled at the times t (descending, t[0] first).  Integrated in ln t.

    Returns a dict of arrays on the grid t:
        t, tau (proper time since t[0]), w (n, 3), gamma = sqrt(1 + w.w), speed = sqrt(1 - 1/gamma^2)
    Note that as the singularity is approached the axes with positive exponent contract, w_i ~ 1/a_i grows,
    gamma diverges and the observer's proper time per unit t shrinks: the clock slows relative to t.
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) >= 0.0):
        raise ValueError("t must be a descending 1-D array with at least two points")
    if not (mm.t_end <= t[-1] and t[0] <= mm.t0):
        raise ValueError("t must lie inside the solution's range")

    def rhs(s, y):
        tt = np.exp(s)
        x, v = mm.state(tt)
        dw_dtau, dt_dtau = geodesic_frame_rhs(y[:3], np.exp(x[0]), v[0] / tt, branch=-1)
        dtau_ds = tt / dt_dtau                                  # dt/ds = t, dt/dtau = -gamma
        return np.concatenate([dw_dtau * dtau_ds, [dtau_ds]])

    sol = solve_ivp(rhs, [np.log(t[0]), np.log(t[-1])], np.concatenate([np.asarray(w0, float), [0.0]]),
                    t_eval=np.log(t), method="DOP853", rtol=1e-10, atol=1e-13)
    if not sol.success:
        raise RuntimeError(f"geodesic integration failed: {sol.message}")
    w = sol.y[:3].T
    gamma = np.sqrt(1.0 + np.sum(w ** 2, axis=1))
    return dict(t=t, tau=sol.y[3], w=w, gamma=gamma, speed=np.sqrt(np.clip(1.0 - 1.0 / gamma ** 2, 0.0, 1.0)))


# --------------------------------------------------------------------------
#  Sensor riding through the bounces
# --------------------------------------------------------------------------

def mixmaster_sensor_run(mm: Mixmaster, t_start: float, t_stop: float, rate_scale: Optional[float] = None,
                         cfg: Optional[Config] = None, K_ref: Optional[float] = None,
                         K_trans: Optional[float] = None, v_ref: float = 0.5,
                         steps_per_efold: float = 25.0, periods_per_step: float = 0.03,
                         phase_per_step: float = 0.3, w0: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """
    Jaynes-Cummings sensor on a comoving observer of `mm`, from time-to-singularity
    t_start down to t_stop (t_start > t_stop), riding through whatever bounce lies
    between them.  The observer ages tau = t_start - t (synchronous gauge, lapse 1).

    Time scales.  The bounces of a Mixmaster solution are separated by tens of e-folds
    of t, so a sensor with fixed frequencies can only resolve one at a time.  `rate_scale`
    multiplies all of the sensor's rates (omega_c, omega_a, g, kappa, gamma), i.e. sets
    its clock in units of the local curvature time; default 3 / sqrt(t_start t_stop).
    The time step is the smallest of: `periods_per_step` sensor periods, `1/steps_per_efold`
    of the remaining t, and `phase_per_step` radians of the (curvature-dependent) cavity phase.

    The curvature scale is the Weyl magnitude sqrt(K^2 + K*^2) (== K in a Kasner epoch; K itself
    changes sign at a bounce).  By default K_ref = K_trans = sqrt(K(t_start) K(t_stop)),
    so rho_hat = 1 and C_Q = 1/2 mid-window.  The potential term is zero and the observer is at
    rest relative to the comoving frame (speed 0): modelling choices, as in qgl_curvature.

    Moving observer.  With `w0` (frame 4-velocity components at t_start, see `mixmaster_geodesic`) the sensor
    rides a geodesic instead of the comoving worldline.  Curvature scalars depend only on t, so K(t) is the same;
    what changes is (i) proper time: dtau = -dt / gamma, so the sensor ages less than the comoving clock, and
    increasingly so as gamma grows toward the singularity, and (ii) the speed, which enters the sensor
    coupling g through s_v.  Extra columns: gamma, w1, w2, w3.
    Columns: tau, t, K (the magnitude), C_Q, rho_hat, speed, omega_c, n, sigma_z, c, purity,
    K_signed, K_star, curv_term.
    """
    if not (mm.t_end <= t_stop < t_start <= mm.t0):
        raise ValueError("need t_end <= t_stop < t_start <= t0")
    cfg = copy.deepcopy(cfg) if cfg is not None else Config()
    rs = float(rate_scale if rate_scale is not None else 3.0 / np.sqrt(t_start * t_stop))
    for name in ("omega_cavity_0", "omega_atom_0", "g_0", "kappa_0", "gamma_0"):
        setattr(cfg, name, getattr(cfg, name) * rs)

    K_lo, K_hi = float(mm.weyl_magnitude(t_start)[0]), float(mm.weyl_magnitude(t_stop)[0])
    K_geo = float(np.sqrt(K_lo * K_hi))
    K_ref = float(K_ref if K_ref is not None else K_geo)
    K_trans = float(K_trans if K_trans is not None else K_geo)
    omega_unit = 2.0 * np.pi * rs

    # adaptive proper-time grid; t is advanced by exact decrements so tau stays precise
    ts, K = [float(t_start)], [K_lo]
    while ts[-1] > t_stop:
        rho_hat = float(qc.density_proxy(K[-1], K_ref))
        omega_c = cfg.omega_cavity_0 * (1.0 + cfg.s_rho * rho_hat)
        dt = min(periods_per_step / rs, ts[-1] / steps_per_efold, phase_per_step / omega_c)
        t_next = max(ts[-1] - dt, t_stop)
        ts.append(t_next)
        K.append(float(mm.weyl_magnitude(t_next)[0]))
    ts, K = np.array(ts), np.array(K)
    if w0 is None:
        tau, speed = float(t_start) - ts, np.zeros_like(ts)
    else:
        geo = mixmaster_geodesic(mm, w0, ts)
        tau, speed = geo["tau"], geo["speed"]

    df = qc.sensor_run_along(tau, K, speed, cfg=cfg, K_ref=K_ref, K_trans=K_trans, v_ref=v_ref)
    df.insert(1, "t", ts)
    if w0 is not None:
        df["gamma"] = geo["gamma"]
        for i in range(3):
            df[f"w{i + 1}"] = geo["w"][:, i]
    df["K_signed"], df["K_star"] = mm.weyl_invariants(ts)
    df["curv_term"] = mm.potential_fraction(ts)
    return df


def main() -> None:
    u0 = 3.7
    mm = Mixmaster(u0, t0=1e6, t_bounce=1.0, t_end=1e-60)
    print(f"Mixmaster from u = {u0} (BKL u-sequence {[round(e['u'], 3) for e in era_sequence(u0, 6)]})")
    print("\nKasner epochs found in the numerical solution (exponents per axis) vs the Landau-Lifshitz map:")
    print(f"{'epoch':>5} {'t_start':>10} {'ln t span':>10}   {'measured v':<30} predicted")
    p = np.array(qk.kasner_exponents(u0))
    for i, e in enumerate(mm.epochs(n_grid=60000, threshold=2e-3)):
        print(f"{i:5d} {e['t_start']:10.3g} {e['dln_t']:10.2f}   {str(np.round(e['v'], 4)):<30} {np.round(p, 4)}")
        p = bkl_transition(p)
    tb = mm.bounce_times()
    print(f"\nbounces at t = {np.array2string(tb, precision=3)}; max |constraint| = "
          f"{np.max(np.abs(mm.constraint(np.geomspace(mm.t0, mm.t_end, 400)))):.1e}")

    t_p = float(tb[0])
    df = mixmaster_sensor_run(mm, 8.0 * t_p, t_p / 8.0)
    print(f"\nSensor through the first bounce (t = {t_p:.3g}), window t = {8 * t_p:.3g} -> {t_p / 8:.3g}:")
    print(df.iloc[:: max(1, len(df) // 10)][["tau", "t", "K", "K_signed", "C_Q", "rho_hat", "omega_c", "n", "sigma_z"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
