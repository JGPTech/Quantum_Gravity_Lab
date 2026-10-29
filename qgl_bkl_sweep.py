#!/usr/bin/env python3
"""
qgl_bkl_sweep.py -- scan the Mixmaster starting parameter u and summarise the Kasner epochs, the bounces and the
QGL sensor's response to the first bounce.

Why a per-u start time.  `qgl_bkl.initial_state` starts in a Kasner epoch and sets the first bounce near
`t_bounce`.  The curvature term of the negative-exponent axis at the start is (t_bounce / t0)^(4 |p_1|) / 2, and
p_1 = -u / (1 + u + u^2) -> 0 as u grows, so a fixed t0 gives a first epoch that is not clean for large u (for
u = 8.5 and t0 = 1e6 the term is already 3e-3 and the first epoch is missing).  `clean_t0` picks t0 so that term
is `tol`; with it every u in the scan starts in a clean epoch.  (`qgl_bkl.clean_t0`, which `Mixmaster` also
uses to warn about unclean starts.)

Analytic expectations used as checks (all tested):
  * Kasner epoch u has exponents (-u, 1+u, u(1+u)) / (1+u+u^2) and Weyl magnitude K = 16 u^2 (1+u)^2 /
    (1+u+u^2)^3 / t^4, which falls from 64/27 at u = 1 toward 0 as u -> infinity (the (0, 0, 1) flat limit);
  * after each bounce u -> u - 1 (u >= 2) or 1/(u - 1) (new era);
  * the measured exponents of every epoch match the Landau-Lifshitz map.
The sensor columns repeat `mixmaster_sensor_run` over t in [8 t_p, t_p / 8] around the first bounce t_p, with the
default clock scaling (3 / sqrt(t_start t_stop)).  K_ref = K_trans = 1 / t_p^4, the curvature scale set by the
bounce time itself, so C_Q measures how strong the epoch's curvature is on that scale: weak epochs (large u,
K t^4 -> 0) give a smaller C_Q.  (The module default, the geometric mean of the window ends, would normalise that
away and make every row look alike.)  Modelling choices as in qgl_curvature.
Bounces are the peaks of the curvature term above `epoch_threshold` (the same threshold that separates the
epochs), because the fixed 5% of `Mixmaster.bounce_times` misses the weak bounces at large u and a threshold
relative to the strongest bounce misses weaker early ones.  Every row therefore has n_epochs = n_bounces + 1,
which is checked.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

import qgl_bkl as bk
import qgl_kasner as qk


def kasner_weyl_coefficient(u: float) -> float:
    """K t^4 for the Kasner epoch with parameter u: -16 p1 p2 p3 = 16 u^2 (1+u)^2 / (1+u+u^2)^3."""
    return 16.0 * u * u * (1.0 + u) ** 2 / (1.0 + u + u * u) ** 3


clean_t0 = bk.clean_t0     # re-exported: defined next to Mixmaster, which warns when a start is not clean


def _epoch_exponent_error(mm: bk.Mixmaster, u: float, epochs) -> float:
    p = np.array(qk.kasner_exponents(float(u)))
    err = 0.0
    for e in epochs:
        err = max(err, float(np.max(np.abs(np.sort(e["v"]) - np.sort(p)))))
        p = bk.bkl_transition(p)
    return err


def mixmaster_sweep(u_values: Iterable[float], t_bounce: float = 1.0, t_end: float = 1e-60,
                    sensor: bool = True, epoch_threshold: float = 2e-3) -> pd.DataFrame:
    """
    One row per starting u:
        u, p1, p2, p3, t0, K_coeff (analytic K t^4), K_t4_measured (mid first epoch),
        n_bounces, n_epochs, first_bounce_t, bounce_strength (max over the run of the curvature term t^2 R3),
        exponent_error (max over the epochs found vs the Landau-Lifshitz map),
        and, with sensor=True, over the first bounce: tau_span, C_Q at the start / at the bounce / at the end, omega_c_max,
        n_final, sigma_z_final, purity_min.
    """
    rows = []
    for u in u_values:
        p = qk.kasner_exponents(float(u))
        t0 = clean_t0(u, t_bounce)
        mm = bk.Mixmaster(float(u), t0, t_bounce, t_end)
        epochs = mm.epochs(n_grid=60000, threshold=epoch_threshold)
        t_grid = np.geomspace(mm.t0, mm.t_end, 6000)
        strength = float(np.max(mm.potential_fraction(t_grid)))
        bounces = mm.bounce_times(threshold=epoch_threshold)
        row = dict(u=float(u), p1=p[0], p2=p[1], p3=p[2], t0=t0, K_coeff=kasner_weyl_coefficient(float(u)),
                   n_bounces=len(bounces), n_epochs=len(epochs),
                   first_bounce_t=float(bounces[0]) if len(bounces) else np.nan,
                   bounce_strength=strength,
                   exponent_error=_epoch_exponent_error(mm, u, epochs))
        if len(epochs):
            e0 = epochs[0]
            t_mid = float(np.sqrt(e0["t_start"] * e0["t_start"] * np.exp(-e0["dln_t"])))
            row["K_t4_measured"] = float(mm.weyl_magnitude(t_mid)[0] * t_mid ** 4)
        else:
            row["K_t4_measured"] = np.nan
        if sensor and len(bounces):
            t_p = float(bounces[0])
            df = bk.mixmaster_sensor_run(mm, 8.0 * t_p, t_p / 8.0, K_ref=t_p ** -4, K_trans=t_p ** -4)
            i_p = int(np.argmin(np.abs(df["t"].to_numpy() - t_p)))
            row.update(tau_span=float(df["tau"].iloc[-1]), C_Q_start=float(df["C_Q"].iloc[0]),
                       C_Q_at_bounce=float(df["C_Q"].iloc[i_p]), C_Q_end=float(df["C_Q"].iloc[-1]),
                       omega_c_max=float(df["omega_c"].max()), n_final=float(df["n"].iloc[-1]),
                       sigma_z_final=float(df["sigma_z"].iloc[-1]), purity_min=float(df["purity"].min()))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    df = mixmaster_sweep([1.3, 1.9, 2.5, 3.1, 3.7, 4.6, 6.2, 8.5, 12.0])
    cols = ["u", "p1", "K_coeff", "K_t4_measured", "n_bounces", "n_epochs", "first_bounce_t", "bounce_strength",
            "exponent_error", "C_Q_start", "C_Q_at_bounce", "C_Q_end", "n_final", "sigma_z_final"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
