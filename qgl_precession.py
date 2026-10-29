#!/usr/bin/env python3
"""
qgl_precession.py -- event-based perihelion-precession measurement for QGL.

`calibrate.measure_precession_error` finds periapsis as local minima of r(t) on
a uniform output grid, so its angle resolution is set by the output spacing.
At modest `n_output` that resolution is comparable to the precession itself
(~1e-3 rad/orbit), and the result can even have the wrong sign.

This module integrates the same equations of motion
(`unified_sandbox_core.unified_equations_of_motion`) but locates every
periapsis with an ODE *event* (r . v = 0, crossing from negative to positive),
so each periapsis angle is accurate to the integrator tolerance.  The
orbit-to-orbit scatter of the shift is then a direct, quotable error bar.

Public functions
----------------
configure_orbit(cfg, p, e)      refresh p, e and everything derived from them
measure_precession(cfg, ...)    -> dict with shift, theory, ratio, scatter
solve_complexity_power(cfg,...) -> complexity_power giving ratio == 1 (or None)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from unified_sandbox_core import (
    Config,
    canonical_periapsis_ic,
    unified_equations_of_motion,
    k_rq_from_pe,
)


def configure_orbit(cfg: Config, p: float, e: float) -> Config:
    """Set (p, e) and refresh alpha_GR, k_rq (dynamic mode) and r_q_transition."""
    cfg.p = float(p)
    cfg.e = float(e)
    if getattr(cfg, "k_rq_ratio_mode", "fixed") == "dynamic":
        cfg.k_rq_ratio = k_rq_from_pe(cfg.p, cfg.e, default=cfg.k_rq_ratio_default)
    cfg.alpha_gr = 3.0 * cfg.G * cfg.M * (cfg.G * cfg.M * cfg.p) / cfg.c ** 2
    cfg.update_r_q_transition_from_orbit()
    return cfg


def gr_1pn_shift(cfg: Config) -> float:
    """1PN perihelion advance per orbit, 6 pi G M / (c^2 p), in radians."""
    return 6.0 * np.pi * cfg.G * cfg.M / (cfg.c ** 2 * cfg.p)


def measure_precession(cfg: Config, n_orbits: int = 6,
                       rtol: float = 1e-11, atol: float = 1e-13) -> dict:
    """
    Integrate `n_orbits` Keplerian periods from periapsis and return

        shift   mean periapsis advance per orbit beyond 2 pi   [rad]
        scatter standard deviation of the per-orbit advances    [rad]
        theory  1PN GR value                                    [rad]
        ratio   shift / theory
        n_peri  number of periapsis events found (n_peri - 1 intervals averaged)
        ok      False if fewer than two periapsis events were found
    """
    r0, v0 = canonical_periapsis_ic(cfg)
    a = cfg.p / (1.0 - cfg.e ** 2)
    period = 2.0 * np.pi * np.sqrt(a ** 3 / cfg.G)

    def periapsis(_t, y):
        return y[0] * y[3] + y[1] * y[4] + y[2] * y[5]
    periapsis.direction = 1.0

    sol = solve_ivp(
        lambda t, y: unified_equations_of_motion(t, y, cfg),
        [0.0, n_orbits * period],
        list(r0) + list(v0),
        events=periapsis, method="DOP853", rtol=rtol, atol=atol,
    )
    ev = sol.y_events[0]
    theory = gr_1pn_shift(cfg)
    if len(ev) < 2:
        return dict(shift=np.nan, scatter=np.nan, theory=theory, ratio=np.nan,
                    n_peri=len(ev), ok=False)

    ang = np.unwrap(np.arctan2(ev[:, 1], ev[:, 0]))
    d = np.diff(ang)
    # wrap each interval to (-pi, pi]: the advance beyond a full turn
    d = (d + np.pi) % (2.0 * np.pi) - np.pi
    shift = float(d.mean())
    return dict(shift=shift, scatter=float(d.std()), theory=theory,
                ratio=shift / theory if theory else np.nan,
                n_peri=int(len(ev)), ok=True)


def solve_complexity_power(cfg: Config, lo: float = 0.05, hi: float = 4.0,
                           n_orbits: int = 4, xtol: float = 1e-6):
    """
    Find complexity_power such that the measured precession equals 1PN GR.

    Uses brentq on ratio(p) - 1, which needs the root bracketed by [lo, hi];
    returns None when it is not (the shift is not reachable by that knob for
    this orbit).  Leaves cfg.complexity_power set to the value found, or
    restored on failure.
    """
    saved = cfg.complexity_power

    def f(cp):
        cfg.complexity_power = float(cp)
        return measure_precession(cfg, n_orbits=n_orbits)["ratio"] - 1.0

    try:
        flo, fhi = f(lo), f(hi)
        if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
            cfg.complexity_power = saved
            return None
        root = brentq(f, lo, hi, xtol=xtol)
        cfg.complexity_power = float(root)
        return float(root)
    except Exception:
        cfg.complexity_power = saved
        return None


if __name__ == "__main__":
    c = configure_orbit(Config(), 100.0, 0.6)
    print(measure_precession(c))
