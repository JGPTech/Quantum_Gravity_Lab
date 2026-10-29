#!/usr/bin/env python3
"""
qgl_curvature.py -- drive QGL's complexity scalar and Jaynes-Cummings sensor from
the Kretschmann curvature scalar instead of from the radius.

QGL's core is written in terms of r:

    C_Q(r)     = 1 / (1 + (r / r_trans)^p)                    complexity scalar
    rho(r)     ~ r^-2                                         density proxy for omega_c

For a Schwarzschild background the Kretschmann scalar is
K(r) = 48 (GM)^2 / (c^4 r^6), so both are *already* functions of curvature:

    (r / r_trans)^p      = (K_trans / K)^(p/6)
    rho / rho_ref        = (K / K_ref)^(1/3)

`complexity_from_curvature` and `density_proxy` are those expressions written in
K.  On Schwarzschild they reproduce the core's numbers (tests/test_curvature.py),
and unlike the r-forms they apply to any spacetime whose K is known -- here the
Kasner cosmology (qgl_kasner), where there is no radius and no Newtonian
potential at all.

Sensor coupling in a general spacetime
--------------------------------------
`jc_params_from_curvature` mirrors `unified_sandbox_core.jc_params_from_classical`
term for term, with rho_hat taken from K.  The potential term has no meaning in
Kasner and defaults to zero; the velocity term uses the speed relative to the
comoving observers.  The sensor evolves in its own proper time.  Those choices
are modelling decisions, not derived physics.

The sensor uses `LindbladJCEngine` (positivity-preserving decay), as the tokamak
mode does; the core engine is not used because its decay step breaks positivity.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from unified_sandbox_core import Config
from qgl_jc_lindblad import LindbladJCEngine
import qgl_kasner as qk


# --------------------------------------------------------------------------
#  Kretschmann scalars
# --------------------------------------------------------------------------

def schwarzschild_kretschmann(r, cfg: Config):
    """K = 48 (GM)^2 / (c^4 r^6), the Schwarzschild Kretschmann scalar in QGL units."""
    return 48.0 * (cfg.G * cfg.M) ** 2 / (cfg.c ** 4 * np.asarray(r, dtype=float) ** 6)


def kretschmann_of_diagonal_metric(diag_entries, coords) -> "object":
    """
    Exact Kretschmann scalar R_abcd R^abcd of a diagonal metric, computed with
    EinsteinPy's symbolic tensors.  `diag_entries` are sympy expressions in `coords`.
    """
    import sympy as sp
    from einsteinpy.symbolic import ChristoffelSymbols, MetricTensor, RiemannCurvatureTensor

    g = sp.diag(*diag_entries)
    metric = MetricTensor(g.tolist(), list(coords), config="ll")
    riem = RiemannCurvatureTensor.from_christoffels(ChristoffelSymbols.from_metric(metric))
    r_llll = riem.change_config("llll").tensor()
    n = len(diag_entries)
    total = 0
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    v = r_llll[a, b, c, d]
                    if v != 0:
                        total += v * v / (g[a, a] * g[b, b] * g[c, c] * g[d, d])
    return sp.simplify(total)


# --------------------------------------------------------------------------
#  Complexity scalar and density proxy from K
# --------------------------------------------------------------------------

def complexity_from_curvature(K, K_trans: float, power: float):
    """
    C_Q = 1 / (1 + (K_trans / K)^(power / 6)).

    The exponent power/6 is what makes this identical to the core's
    1 / (1 + (r / r_trans)^power) on Schwarzschild, where K ~ r^-6.  Large
    curvature gives C_Q -> 1, flat space gives 0.
    """
    K = np.maximum(np.asarray(K, dtype=float), 0.0)
    with np.errstate(divide="ignore"):
        ratio = np.where(K > 0.0, (K_trans / np.where(K > 0.0, K, 1.0)) ** (power / 6.0), np.inf)
    return 1.0 / (1.0 + ratio)


def density_proxy(K, K_ref: float):
    """rho_hat = (K / K_ref)^(1/3): equals rho(r)/rho_ref with rho ~ r^-2 on Schwarzschild."""
    return (np.maximum(np.asarray(K, dtype=float), 0.0) / K_ref) ** (1.0 / 3.0)


def jc_params_from_curvature(K: float, v_mag: float, cfg: Config, K_ref: float,
                             v_ref: float, phi_hat: float = 0.0) -> Dict[str, float]:
    """
    JC cavity parameters from local curvature K, speed v_mag and (optional)
    normalised potential phi_hat; the same couplings as `jc_params_from_classical`.
    """
    rho_hat = float(density_proxy(K, K_ref))
    v_hat = v_mag / (v_ref + 1e-12)
    return {
        "omega_c": float(cfg.omega_cavity_0 * (1.0 + cfg.s_rho * rho_hat)),
        "omega_a": float(cfg.omega_atom_0 * (1.0 + cfg.s_phi * phi_hat)),
        "g": float(cfg.g_0 * (1.0 + cfg.s_v * v_hat)),
        "kappa": float(cfg.kappa_0 * (1.0 + cfg.s_kappa * rho_hat)),
        "gamma": float(cfg.gamma_0),
    }


# --------------------------------------------------------------------------
#  Sensor riding a Kasner worldline
# --------------------------------------------------------------------------

def kasner_sensor_run(p: Sequence[float], k: Sequence[float], t0: float, t_end: float,
                      cfg: Optional[Config] = None, n_steps: int = 400,
                      t_trans: Optional[float] = None, v_ref: float = 0.5) -> pd.DataFrame:
    """
    Evolve the JC sensor along a Kasner geodesic and return a DataFrame with

        tau, t, K, C_Q, rho_hat, speed, omega_c, n, sigma_z, c, purity

    K_ref is K at t0 (so rho_hat starts at 1).  `t_trans` sets where C_Q = 1/2
    (default: geometric mean of t0 and t_end).  Set t_end < t0 to run toward the
    singularity, where K grows without bound.
    """
    cfg = cfg or Config()
    geo = qk.kasner_geodesic(p, k, t0, t_end, n_out=n_steps)
    K = qk.kretschmann_closed_form(p, geo["t"])
    if not np.all(K > 0.0):
        raise ValueError("Kasner exponents give zero curvature (flat case); the sensor sees no K")
    K_ref = float(qk.kretschmann_closed_form(p, t0))
    t_trans = t_trans if t_trans is not None else float(np.sqrt(t0 * t_end))
    K_trans = float(qk.kretschmann_closed_form(p, t_trans))

    df = sensor_run_along(geo["tau"], K, geo["speed"], cfg=cfg, K_ref=K_ref, K_trans=K_trans, v_ref=v_ref)
    df.insert(1, "t", geo["t"])
    return df


def sensor_run_along(tau, K, speed, cfg: Optional[Config] = None, K_ref: float = 1.0,
                     K_trans: float = 1.0, v_ref: float = 0.5) -> pd.DataFrame:
    """
    Evolve the JC sensor along any worldline given its proper time tau, the Kretschmann
    scalar K(tau) and the speed relative to the local rest frame, and return a DataFrame with

        tau, K, C_Q, rho_hat, speed, omega_c, n, sigma_z, c, purity

    The sensor takes a proper-time step between successive samples with the couplings
    evaluated at the later sample.  K_ref sets rho_hat = 1; K_trans sets C_Q = 1/2.
    """
    cfg = cfg or Config()
    engine = LindbladJCEngine(cfg)
    rows = []
    for i in range(len(tau)):
        params = jc_params_from_curvature(K[i], speed[i], cfg, K_ref, v_ref)
        if i > 0:
            engine.evolve(params, tau[i] - tau[i - 1])
        ex = engine.compute_expectations(params)
        rows.append(dict(tau=tau[i], K=K[i],
                         C_Q=float(complexity_from_curvature(K[i], K_trans, cfg.complexity_power)),
                         rho_hat=float(density_proxy(K[i], K_ref)), speed=speed[i],
                         omega_c=params["omega_c"], n=ex["n"], sigma_z=ex["sigma_z"], c=ex["c"],
                         purity=float(np.real(np.trace(engine.rho @ engine.rho)))))
    return pd.DataFrame(rows)


def main() -> None:
    p = qk.kasner_exponents(1.0)
    df = kasner_sensor_run(p, k=(0.3, -0.2, 0.5), t0=10.0, t_end=2.0, n_steps=200)
    print(f"Kasner p = {tuple(round(float(x), 4) for x in p)}, contracting from t=10 to t=2")
    print(df.iloc[::40][["tau", "t", "K", "C_Q", "rho_hat", "speed", "omega_c", "n", "sigma_z"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
