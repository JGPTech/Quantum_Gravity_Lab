#!/usr/bin/env python3
"""
qgl_compare.py -- the cosmologies in this repo side by side, on one footing.

Scenarios (each a comoving observer's worldline through a window of its history):

    kasner      vacuum Kasner (u = 2), contracting toward the singularity
    mixmaster   vacuum Bianchi IX, around the first bounce
    pancake     the non-singular pancake of Dechant-Lasenby-Hobson (qgl_pancake), through t = 0
    bounce      the dust bounce toy with maximum density (qgl_flrw), through t = 0
    lcdm        flat LCDM, a = 0.1 -> 1
    toy         the sigmoid "entropy" dark-term toy (qgl_flrw), a = 0.1 -> 1
    ccbh        the coupled-black-hole conversion toy (qgl_ccbh), a = 0.1 -> 1

Common footing.  Every scenario supplies the curvature magnitude |K|(tau) along the observer's own proper time
and one fiducial event: the bounce / the pancake / today / the first bounce of Mixmaster / the start of the
Kasner window.  Then

    K_ref = |K| at the fiducial event,   T_K = K_ref^(-1/4)   (the curvature time),
    K_hat = |K| / K_ref ,                tau_hat = (tau - tau_fiducial) / T_K ,

so every scenario has K_hat = 1 at tau_hat = 0, and the sensor sees the same couplings everywhere: the default
`Config`, rates scaled by `rate_scale` per curvature time, K_ref = K_trans = K_ref above (so C_Q = 1/2 at the
fiducial event).  Differences in the sensor's response are then differences of the curvature history alone.

What this does and does not show.  |K| is used throughout (K is not sign-definite: Mixmaster's K changes sign at a
bounce and the pancake's does far from it); that discards the sign.  The windows are chosen by hand and are not
equal in how much curvature they span (K_hat_max / K_hat_min is reported).  The sensor coupling is a modelling
choice (qgl_curvature), not derived physics.  The flat-FLRW scenarios are toys, not fits to data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from unified_sandbox_core import Config
from qgl_jc_lindblad import LindbladJCEngine
import qgl_curvature as qc


@dataclass
class Scenario:
    name: str
    tau: np.ndarray              # proper time along the observer, native units, ascending
    K: np.ndarray                # curvature magnitude |K|, native units, same length
    i_fid: int                   # index of the fiducial event
    singular_history: bool       # does the full history of the model hit a curvature singularity?
    note: str = ""

    @property
    def K_ref(self) -> float:
        return float(self.K[self.i_fid])

    @property
    def T_K(self) -> float:
        return self.K_ref ** -0.25

    def normalised(self):
        """(tau_hat, K_hat): dimensionless, K_hat = 1 and tau_hat = 0 at the fiducial event."""
        return (self.tau - self.tau[self.i_fid]) / self.T_K, self.K / self.K_ref


# --------------------------------------------------------------------------
#  Scenario builders
# --------------------------------------------------------------------------

def kasner_scenario(u: float = 2.0, t_hi: float = 1.0, t_lo: float = 0.05, n: int = 400) -> Scenario:
    import qgl_kasner as qk
    p = qk.kasner_exponents(u)
    t = np.geomspace(t_hi, t_lo, n)
    return Scenario("kasner", t_hi - t, np.abs(qk.kretschmann_closed_form(p, t)), 0, True,
                    f"u = {u}; fiducial = start of the window; K ~ t^-4")


def mixmaster_scenario(u: float = 3.7, n: int = 400) -> Scenario:
    import qgl_bkl as bk
    mm = bk.Mixmaster(u, bk.clean_t0(u, 1.0), 1.0, 1e-60)
    t_p = float(mm.bounce_times()[0])
    t = np.geomspace(8.0 * t_p, t_p / 8.0, n)
    K = np.abs(mm.weyl_magnitude(t))
    i = int(np.argmin(np.abs(t - t_p)))
    return Scenario("mixmaster", 8.0 * t_p - t, K, i, True, f"u = {u}; Weyl magnitude around the first bounce")


def pancake_scenario(a0=1.0, b0=3.0, f0=2.0, m=0.5, Lam=0.0, half_window=3.0, n: int = 400) -> Scenario:
    import qgl_pancake as qp
    sol = qp.pancake_solution(a0, b0, f0, m, Lam, t_max=half_window, n=n)
    K = qp.pancake_kretschmann(sol)
    t = sol["t"]
    K0 = qp.kretschmann_at_pancake(b0, f0, m, Lam, 1.0, a0)
    neg, pos = t < 0, t > 0
    tt = np.concatenate([t[neg], [0.0], t[pos]])
    KK = np.abs(np.concatenate([K[neg], [K0], K[pos]]))
    return Scenario("pancake", tt - tt[0], KK, int(np.flatnonzero(tt == 0.0)[0]), False,
                    "volume -> 0 at t = 0, curvature finite; fiducial = the pancake")


def bounce_scenario(Om=0.3, rho_c=10.0, half_window=2.0, n: int = 400) -> Scenario:
    import qgl_flrw as fl
    t = np.concatenate([-np.geomspace(half_window, 1e-3 * half_window, n // 2)[:-1], [0.0],
                        np.geomspace(1e-3 * half_window, half_window, n // 2)[1:]])
    K = fl.flrw_kretschmann(fl.bounce_hubble(t, Om, rho_c), fl.bounce_hubble_dot(t, Om, rho_c))
    return Scenario("bounce", t - t[0], np.abs(K), int(np.flatnonzero(t == 0.0)[0]), False,
                    "dust with maximum density; fiducial = the bounce")


def _flrw_scenario(name: str, E2: Callable, note: str, a_lo=0.1, a_hi=1.0, n: int = 300, singular=True) -> Scenario:
    import qgl_flrw as fl
    a = np.geomspace(a_lo, a_hi, n)
    h = fl.flrw_history(E2, a)
    return Scenario(name, h["t"].to_numpy() - h["t"].iloc[0], np.abs(h["K"].to_numpy()), n - 1, singular, note)


def lcdm_scenario(**kw) -> Scenario:
    import qgl_flrw as fl
    return _flrw_scenario("lcdm", lambda x: fl.lcdm_E2(x, 0.3), "flat LCDM, Om = 0.3; fiducial = today", **kw)


def toy_scenario(**kw) -> Scenario:
    import qgl_flrw as fl
    p = fl.ToyParams(0.3, 0.0, 0.4, 4.0)
    return _flrw_scenario("toy", lambda x: fl.toy_E2(x, p), "sigmoid dark term (a_t = 0.4, k = 4); fiducial = today", **kw)


def ccbh_scenario(eps: float = 1.0, **kw) -> Scenario:
    import qgl_ccbh as cc
    bg = cc.ccbh_background(0.3, eps, 3.0)
    return _flrw_scenario("ccbh", bg.E2, f"coupled black holes, eps = {eps}, k = 3; fiducial = today", **kw)


def all_scenarios() -> List[Scenario]:
    return [kasner_scenario(), mixmaster_scenario(), pancake_scenario(), bounce_scenario(),
            lcdm_scenario(), toy_scenario(), ccbh_scenario()]


# --------------------------------------------------------------------------
#  Sensor on the common footing
# --------------------------------------------------------------------------

def sensor_on(sc: Scenario, rate_scale: float = 4.0, cfg: Optional[Config] = None, phase_per_step: float = 0.3,
              max_step: float = 0.25, max_points: int = 6000) -> pd.DataFrame:
    """
    The JC sensor along `sc` in dimensionless time, with K_ref = K_trans = 1 (so C_Q = 1/2 at the fiducial event).
    The sensor clock runs at `rate_scale` sensor units per curvature time.  The step is the smaller of `max_step`
    curvature times and `phase_per_step` radians of the cavity phase.  Columns: tau_hat, K_hat, C_Q, rho_hat,
    omega_c, n, sigma_z, c, purity.
    """
    cfg = cfg or Config()
    tau_hat, K_hat = sc.normalised()
    logK = np.log(np.maximum(K_hat, 1e-300))
    grid = [float(tau_hat[0])]
    while grid[-1] < tau_hat[-1] and len(grid) < max_points:
        k = float(np.exp(np.interp(grid[-1], tau_hat, logK)))
        omega = cfg.omega_cavity_0 * (1.0 + cfg.s_rho * float(qc.density_proxy(k, 1.0)))
        grid.append(min(grid[-1] + min(max_step, phase_per_step / (omega * rate_scale)), float(tau_hat[-1])))
    grid = np.array(grid)
    Kg = np.exp(np.interp(grid, tau_hat, logK))
    df = qc.sensor_run_along((grid - grid[0]) * rate_scale, Kg, np.zeros_like(grid), cfg=cfg, K_ref=1.0, K_trans=1.0)
    df.insert(0, "tau_hat", grid)
    df.insert(1, "K_hat", Kg)
    return df.drop(columns=["tau", "K", "speed"])


def summarise(sc: Scenario, df: pd.DataFrame) -> Dict[str, float]:
    tau_hat, K_hat = sc.normalised()
    dlnK = np.abs(np.gradient(np.log(np.maximum(K_hat, 1e-300)), tau_hat))
    return dict(scenario=sc.name, singular_history=sc.singular_history, K_ref=sc.K_ref, T_K=sc.T_K,
                window=float(tau_hat[-1] - tau_hat[0]), K_hat_min=float(K_hat.min()), K_hat_max=float(K_hat.max()),
                max_dlnK_dtau=float(dlnK.max()), C_Q_min=float(df["C_Q"].min()), C_Q_max=float(df["C_Q"].max()),
                omega_c_max=float(df["omega_c"].max()), n_final=float(df["n"].iloc[-1]),
                sigma_z_final=float(df["sigma_z"].iloc[-1]), purity_min=float(df["purity"].min()))


def compare(scenarios: Optional[List[Scenario]] = None, rate_scale: float = 4.0):
    """Run the sensor on every scenario: (summary DataFrame, {name: sensor DataFrame})."""
    scenarios = scenarios or all_scenarios()
    runs = {sc.name: sensor_on(sc, rate_scale) for sc in scenarios}
    table = pd.DataFrame([summarise(sc, runs[sc.name]) for sc in scenarios])
    return table, runs


def plot_comparison(path: str, scenarios: Optional[List[Scenario]] = None, rate_scale: float = 4.0,
                    runs: Optional[Dict[str, pd.DataFrame]] = None) -> None:
    """
    Two panels: K_hat(tau_hat) and the sensor's C_Q(tau_hat), one line per scenario (saved to `path`).  Pass the
    `runs` returned by `compare` to avoid re-running the sensors.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scenarios = scenarios or all_scenarios()
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]   # Okabe-Ito
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for sc, col in zip(scenarios, palette):
        df = runs[sc.name] if runs is not None else sensor_on(sc, rate_scale)
        ax[0].plot(df["tau_hat"], df["K_hat"], color=col, label=sc.name, lw=1.6)
        ax[1].plot(df["tau_hat"], df["C_Q"], color=col, label=sc.name, lw=1.6)
    ax[0].set(yscale="log", xlabel=r"$\hat\tau=(\tau-\tau_{fid})/T_K$", ylabel=r"$|K|/K_{ref}$",
              title="Curvature magnitude in units of its value at the fiducial event")
    ax[1].set(xlabel=r"$\hat\tau$", ylabel=r"$C_Q$", ylim=(0, 1), title="Complexity scalar (C_Q = 1/2 at the fiducial event)")
    for a in ax:
        a.axvline(0.0, color="0.7", lw=0.8)
        a.grid(alpha=0.25)
    ax[0].legend(frameon=False, fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    import sys
    if "--figure" in sys.argv:
        import os
        os.makedirs("figures", exist_ok=True)
        plot_comparison("figures/qgl_compare.png")
        print("wrote figures/qgl_compare.png")
        return
    table, _ = compare()
    cols = ["scenario", "singular_history", "window", "K_hat_min", "K_hat_max", "max_dlnK_dtau", "C_Q_min", "C_Q_max",
            "omega_c_max", "n_final", "sigma_z_final"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
