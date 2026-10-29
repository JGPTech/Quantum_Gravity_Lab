#!/usr/bin/env python3
"""
qgl_einsteinpy.py -- exact-GR reference values from EinsteinPy, for validating QGL.

QGL reproduces GR phenomenologically (a calibrated 1/r^4 force for the 1PN
periapsis advance, a velocity-rotation term for frame dragging).  This module
provides *exact* general-relativity numbers to compare against, so the size of
the approximation can be quantified rather than assumed.

Two independent references are offered for the Schwarzschild periapsis advance:

  * schwarzschild_advance_exact      closed form (elliptic integral), scipy only
  * einsteinpy_periapsis_advance     numerical geodesic from EinsteinPy
                                     (Schwarzschild, or Kerr with `spin`)

and EinsteinPy's symbolic Schwarzschild metric supplies the exact static lapse.

Units
-----
EinsteinPy works in geometric units (G = c = M = 1).  QGL length scales map to
them by dividing by GM/c^2 (`length_unit`), so QGL's default c = 10, p = 100 is
p = 1e4 GM/c^2.  That is a very weak field: EinsteinPy's geodesic integrator is
stable only for steps of order 1 GM/c^2, which makes one orbit ~1e7 steps
(hours).  The geodesic comparisons therefore use `strong_field_config`
(c = 1, p of order 60-400 GM/c^2), where an orbit is a few thousand steps.

What "the same orbit" means
---------------------------
An orbit is specified by its turning points r_p = p/(1+e), r_a = p/(1-e)
(areal radius, in units of GM/c^2).  QGL starts from a Newtonian periapsis
state, which is the same orbit only up to O(1/p) -- the size of the 2PN
correction being probed.  Comparisons should therefore be read as "agrees to
leading order, with a deficit that shrinks as 1/p", not as equality.

Requires EinsteinPy only for `exact_static_lapse` and `einsteinpy_periapsis_advance`
(pip install einsteinpy).  Everything else needs only numpy/scipy.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import warnings

import numpy as np
from scipy.special import ellipk

from unified_sandbox_core import Config
from qgl_precession import (configure_orbit, gr_1pn_shift, measure_precession,
                            solve_complexity_power)


# --------------------------------------------------------------------------
#  Availability and unit mapping
# --------------------------------------------------------------------------

def has_einsteinpy() -> bool:
    """True if EinsteinPy can be imported (without importing it)."""
    return importlib.util.find_spec("einsteinpy") is not None


def _require_einsteinpy() -> None:
    if not has_einsteinpy():
        raise ImportError("EinsteinPy is required for this function: pip install einsteinpy")


def length_unit(cfg: Config) -> float:
    """Geometric length unit GM/c^2 in QGL units (r_geo = r / length_unit)."""
    return cfg.G * cfg.M / cfg.c ** 2


def strong_field_config(p: float, e: float, a_spin: float = 0.0) -> Config:
    """
    QGL Config in units G = M = c = 1 (so lengths are already GM/c^2).

    `k_rq_ratio_mode` is forced to "fixed" so the result does not depend on a
    calibration table being present.  `a_spin` is the dimensionless Kerr spin
    a/M (QGL's `a_spin` is J c / (G M^2)).
    """
    cfg = Config()
    cfg.c = 1.0                       # before configure_orbit: alpha_GR is derived from c
    cfg.k_rq_ratio_mode = "fixed"
    cfg.k_rq_ratio = cfg.k_rq_ratio_default
    configure_orbit(cfg, p, e)
    cfg.a_spin = float(a_spin)
    return cfg


# --------------------------------------------------------------------------
#  Exact static lapse (EinsteinPy symbolic)
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _lapse_function():
    """sqrt(g_tt) of EinsteinPy's symbolic Schwarzschild metric, as f(r) with r_s = 2."""
    _require_einsteinpy()
    import sympy as sp
    from einsteinpy.symbolic import Schwarzschild

    g_tt = Schwarzschild().tensor()[0, 0]
    syms = {s.name: s for s in g_tt.free_symbols}
    return sp.lambdify(syms["r"], sp.sqrt(g_tt.subs(syms["r_s"], 2)), "numpy")


def exact_static_lapse(r_geo):
    """
    Exact d(tau)/dt of a static observer in Schwarzschild, from EinsteinPy's
    metric, at r_geo (in GM/c^2, so the horizon is r_geo = 2).  Requires r_geo > 2.
    """
    r_geo = np.asarray(r_geo, dtype=float)
    if np.any(r_geo <= 2.0):
        raise ValueError("static lapse is real only outside the horizon (r_geo > 2)")
    return _lapse_function()(r_geo)


# --------------------------------------------------------------------------
#  Exact Schwarzschild periapsis advance (closed form)
# --------------------------------------------------------------------------

def _check_bound(p: float, e: float) -> None:
    if not (0.0 <= e < 1.0) or p <= 6.0 + 2.0 * e:
        raise ValueError(f"need 0 <= e < 1 and p > 6 + 2e for a stable bound orbit (got p={p}, e={e})")


def schwarzschild_constants(p: float, e: float) -> dict:
    """
    Specific energy E and angular momentum L (G = c = M = 1) of the bound
    Schwarzschild geodesic with turning points r_p = p/(1+e), r_a = p/(1-e).
    """
    _check_bound(p, e)
    denom = p * (p - 3.0 - e * e)
    return dict(E=float(np.sqrt(((p - 2.0) ** 2 - 4.0 * e * e) / denom)),
                L=float(p / np.sqrt(p - 3.0 - e * e)))


def schwarzschild_advance_exact(p: float, e: float) -> float:
    """
    Exact periapsis advance per radial period beyond 2 pi [rad], Schwarzschild,
    for turning points r_p = p/(1+e), r_a = p/(1-e) in units of GM/c^2.

    With u = 1/r the orbit obeys (du/dphi)^2 = 2 (u-u1)(u-u2)(u-u3) where
    u1 = (1-e)/p, u2 = (1+e)/p are the turning points and u3 = 1/2 - u1 - u2,
    so the radial period in phi is a complete elliptic integral.
    """
    _check_bound(p, e)
    u1, u2 = (1.0 - e) / p, (1.0 + e) / p
    u3 = 0.5 - u1 - u2
    m = (u2 - u1) / (u3 - u1)
    return float(2.0 * np.sqrt(2.0) * ellipk(m) / np.sqrt(u3 - u1) - 2.0 * np.pi)


# --------------------------------------------------------------------------
#  Numerical geodesic from EinsteinPy
# --------------------------------------------------------------------------

def einsteinpy_periapsis_advance(p: float, e: float, spin: float = 0.0,
                                 delta: float = 1.0, intervals: int = 1) -> np.ndarray:
    """
    Periapsis advance per orbit beyond 2 pi [rad] from an EinsteinPy timelike
    geodesic, one entry per periapsis-to-periapsis interval.

    The particle starts at periapsis r_p = p/(1+e) on the equator with the
    Schwarzschild angular momentum L for these turning points and p_r = 0.  For
    spin != 0 (Kerr, a/M = spin; spin > 0 with L > 0 is prograde) the start is
    the same (r_p, L) state, so the orbit shape differs slightly from the
    Schwarzschild one, as it must.

    Periapsis is where p_r crosses zero upward; the angle is interpolated
    linearly between steps.  `delta` is the integrator step in proper time
    [GM/c^3]; 1.0 is accurate to ~1e-4 rad here (0.5 halves the error), and
    values >~ 2 diverge (OverflowError) in this regime.  Cost is roughly
    3-5 ms per step, ~ (intervals + 0.6) * 2 pi a^1.5 / delta steps.
    """
    _require_einsteinpy()
    from einsteinpy.geodesic import Timelike

    consts = schwarzschild_constants(p, e)
    r_p = p / (1.0 + e)
    a = p / (1.0 - e * e)
    steps = int(np.ceil((intervals + 0.6) * 2.0 * np.pi * a ** 1.5 / delta))

    try:
        geo = Timelike(metric="Kerr" if spin else "Schwarzschild",
                       metric_params=(float(spin),),
                       position=[r_p, np.pi / 2.0, 0.0],
                       momentum=[0.0, 0.0, consts["L"]],
                       return_cartesian=False, suppress_warnings=True,
                       steps=steps, delta=delta)
    except OverflowError as exc:
        raise RuntimeError(f"EinsteinPy integrator diverged at delta={delta}; use a smaller step") from exc

    # columns of the returned trajectory: t, r, theta, phi, p_t, p_r, p_theta, p_phi
    _, y = geo.trajectory
    phi, p_r = y[:, 3], y[:, 5]
    k = np.where((p_r[:-1] < 0.0) & (p_r[1:] >= 0.0))[0]
    peri = np.concatenate([[0.0], phi[k] + (0.0 - p_r[k]) / (p_r[k + 1] - p_r[k]) * (phi[k + 1] - phi[k])])
    if len(peri) < intervals + 1:
        raise RuntimeError(f"found {len(peri) - 1} periapsis intervals, wanted {intervals}")
    return np.diff(peri)[:intervals] - 2.0 * np.pi


# --------------------------------------------------------------------------
#  QGL vs exact
# --------------------------------------------------------------------------

def qgl_precession_vs_exact(p: float, e: float, n_orbits: int = 3) -> dict:
    """
    QGL periapsis advance (c = 1, complexity_power solved so QGL matches 1PN
    exactly) against the exact Schwarzschild value for the same (p, e).

    Returns one_pn, exact, exact_over_1pn, qgl_shift, qgl_over_exact, complexity_power.
    """
    cfg = strong_field_config(p, e)
    cp = solve_complexity_power(cfg, n_orbits=n_orbits)
    if cp is None:
        raise RuntimeError(f"complexity_power could not be solved for p={p}, e={e}")
    shift = measure_precession(cfg, n_orbits=n_orbits)["shift"]
    exact = schwarzschild_advance_exact(p, e)
    one_pn = gr_1pn_shift(cfg)
    return dict(p=p, e=e, one_pn=one_pn, exact=exact, exact_over_1pn=exact / one_pn,
                qgl_shift=shift, qgl_over_exact=shift / exact, complexity_power=cp)


def qgl_spin_effect(p: float, e: float, a_spin: float, n_orbits: int = 3) -> float:
    """Change in QGL's periapsis advance [rad] caused by a_spin (prograde for a_spin > 0)."""
    base = measure_precession(strong_field_config(p, e, 0.0), n_orbits=n_orbits)["shift"]
    spun = measure_precession(strong_field_config(p, e, a_spin), n_orbits=n_orbits)["shift"]
    return spun - base


def main() -> None:
    ap = argparse.ArgumentParser(description="QGL precession vs exact Schwarzschild GR")
    ap.add_argument("--p", type=float, nargs="+", default=[60.0, 100.0, 200.0, 400.0])
    ap.add_argument("--e", type=float, nargs="+", default=[0.3, 0.6])
    ap.add_argument("--geodesic", action="store_true",
                    help="also integrate an EinsteinPy geodesic for each (p, e) (slow)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    print(f"{'p':>6} {'e':>5} {'1PN':>9} {'exact':>9} {'exact/1PN':>10} {'QGL/exact':>10}"
          + ("  EinsteinPy" if args.geodesic else ""))
    for e in args.e:
        for p in args.p:
            r = qgl_precession_vs_exact(p, e)
            line = (f"{p:6.0f} {e:5.2f} {r['one_pn']:9.5f} {r['exact']:9.5f} "
                    f"{r['exact_over_1pn']:10.4f} {r['qgl_over_exact']:10.4f}")
            if args.geodesic:
                line += f"  {einsteinpy_periapsis_advance(p, e)[0]:9.5f}"
            print(line)


if __name__ == "__main__":
    main()
