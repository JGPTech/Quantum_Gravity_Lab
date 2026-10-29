#!/usr/bin/env python3
"""
qgl_tidal.py -- tidal tensors for the QGL spacetimes, and what they say about "pancake" singularities.

For comoving observers in a diagonal Bianchi universe  ds^2 = -dt^2 + sum a_i^2 sigma_i^2  the geodesic-deviation
equation in the orthonormal frame is  xi_i'' = (a_i'' / a_i) xi_i .  The tidal eigenvalues are therefore

    lambda_i = a_i'' / a_i        (> 0 stretches along axis i, < 0 compresses)

with trace  sum_i lambda_i = -R_00  (zero in vacuum).  Examples:

    Kasner          lambda_i = p_i (p_i - 1) / t^2     -> diverges as t -> 0 (every vacuum Kasner is a cigar there)
    Schwarzschild   (+2, -1, -1) GM / (c^2 r^3)        -> radial stretch, transverse squeeze
    pancake         finite at t = 0 (qgl_pancake)      -> no diverging tides, although the volume vanishes

The tidal strength T = sqrt(sum lambda_i^2) feeds QGL's complexity scalar exactly as K does.  For a static vacuum
field K = 8 T^2 (Schwarzschild: 8 * 6 (GM)^2/r^6 = 48 (GM)^2/r^6); in general K = 8 (E.E - B.B), so
K = 8 T^2 - 8 B.B with B the magnetic Weyl part, which is nonzero in Mixmaster.  Both statements are tested.
`complexity_from_tidal` is `qgl_curvature.complexity_from_curvature` with K -> T^2.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from unified_sandbox_core import Config
import qgl_curvature as qc
import qgl_pancake as qp


def tidal_from_scale_factors(a, add) -> np.ndarray:
    """lambda_i = a_i'' / a_i for scale factors a and second derivatives add (shape (..., 3))."""
    return np.asarray(add, dtype=float) / np.asarray(a, dtype=float)


def kasner_tidal(p: Sequence[float], t) -> np.ndarray:
    """Comoving tidal eigenvalues of Kasner, p_i (p_i - 1) / t^2, shape (len(t), 3) for array t."""
    p = np.asarray(p, dtype=float)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return (p * (p - 1.0))[None, :] / t[:, None] ** 2


def schwarzschild_tidal(r, cfg: Optional[Config] = None) -> np.ndarray:
    """
    (+2, -1, -1) GM / (c^2 r^3) for a freely falling observer: (radial, transverse, transverse), shape (n, 3).
    These are eigenvalues of R_0i0i in 1/length^2 (so K = 8 sum lambda^2 holds); the Newtonian tidal
    acceleration per unit separation is c^2 times this.
    """
    cfg = cfg or Config()
    r = np.atleast_1d(np.asarray(r, dtype=float))
    base = cfg.G * cfg.M / (cfg.c ** 2 * r ** 3)
    return np.stack([2.0 * base, -base, -base], axis=-1)


def pancake_tidal(sol) -> np.ndarray:
    """Comoving tidal eigenvalues R_i'' / R_i along a `qgl_pancake.pancake_solution`, shape (n, 3)."""
    return tidal_from_scale_factors(sol["R"], sol["Rdd"])


def pancake_tidal_at_zero(b0: float, f0: float, m: float, Lam: float = 0.0, kappa: float = 1.0) -> np.ndarray:
    """
    Tidal eigenvalues at the pancake itself: (4/b0^2, e, e) with
    e = R_2''/R_2 = 2 b2 / b0 = Lam/2 + kappa m^2 f0^2/4 - 2/b0^2.
    Finite for every choice of the free data.
    """
    c = qp.pancake_series(1.0, b0, f0, m, Lam, kappa)
    e = 2.0 * c["b2"] / b0
    return np.array([4.0 / b0 ** 2, e, e])


def tidal_strength(lam) -> np.ndarray:
    """T = sqrt(sum_i lambda_i^2); a frame-independent size of the tidal tensor."""
    return np.sqrt(np.sum(np.asarray(lam, dtype=float) ** 2, axis=-1))


def complexity_from_tidal(T, T_trans: float, power: float):
    """
    C_Q = 1 / (1 + (T_trans / T)^(power / 3)): the curvature form with K -> T^2, so it reproduces the core's
    1 / (1 + (r/r_trans)^power) on Schwarzschild, where T ~ r^-3.
    """
    return qc.complexity_from_curvature(np.asarray(T, dtype=float) ** 2, T_trans ** 2, power)


def main() -> None:
    from qgl_kasner import kasner_exponents
    p = kasner_exponents(2.0)
    print("comoving tidal eigenvalues (stretch > 0):")
    for t in (1.0, 0.1, 0.01):
        lam = kasner_tidal(p, t)[0]
        print(f"  Kasner u=2, t={t:<5}: {np.round(lam, 3)}   T = {tidal_strength(lam):.4g}")
    sol = qp.pancake_solution(1.0, 3.0, 2.0, 0.5, t_max=3.0)
    T = tidal_strength(pancake_tidal(sol))
    print(f"  pancake  b0=3        : T(t=0) = {tidal_strength(pancake_tidal_at_zero(3.0, 2.0, 0.5)):.4g},"
          f" max over the solution = {T.max():.4g}")


if __name__ == "__main__":
    main()
