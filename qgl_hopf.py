"""
qgl_hopf.py - the Hopf fibration / Berger sphere reading of the biaxial pancake, on the numerical solution.

The spatial slice of the biaxial Bianchi IX pancake (R_2 = R_3) is S^3 with the metric

    (1/4) [ R_1^2 (d psi + cos(th) d phi)^2 + R_2^2 (d th^2 + sin^2(th) d phi^2) ] ,

i.e. the Hopf fibration S^1 -> S^3 -> S^2 with the fibre scaled by s = R_1 / R_2 (a Berger sphere) and an overall scale
R_2.  Near the pancake s -> 0: the Hopf fibres shrink to zero length while the base S^2 stays at radius R_2 / 2.

This module follows that picture along the real solution of qgl_pancake.pancake_solution (not only its leading order):

  * s(t) = R_1 / R_2,
  * the slice Ricci curvature of the solution against the Berger closed form (2 s^2, 4 - 2 s^2, 4 - 2 s^2) / R_2^2
    (identical by construction of the metric, so this is a check of qgl_bkl.ricci3, not a new result),
  * the winding of a light ray round the fibre, int 2 dt / R_1 / (4 pi) turns, against the leading order
    (2 / (4 pi a0)) ln(t / t_in),
  * the local number of e-folds of t per fibre turn, 2 pi R_1 / t, which tends to 2 pi a0.

What it does not claim: the pancake solution of qgl_pancake is leading order, not a solution of the field equations to all
orders; the S^3 (period 4 pi) identification is checked in tests/test_s3_identifications.py, whether the paper uses S^3
or SO(3) is not.  The Hopf-fibration repository (github.com/brentharts/HopfFibration, hopf_berger.py, hopf_pancake.py)
has the same formulas written independently; its cross-check against this module is listed in the README.

    python3 qgl_hopf.py            print a summary
    python3 qgl_hopf.py --figure   write figures/qgl_hopf.png
"""
from __future__ import annotations

from typing import Dict

import numpy as np

import qgl_horizon as hz
import qgl_pancake as qp

_PAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]   # Okabe-Ito


def berger_parameter(sol: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """t > 0 part of a pancake solution: t, R_1, R_2 and the squashing s = R_1 / R_2 (s = 1 is the round S^3)."""
    pos = sol["t"] > 0
    R = sol["R"][pos]
    return dict(t=sol["t"][pos], R1=R[:, 0], R2=R[:, 1], s=R[:, 0] / R[:, 1])


def berger_ricci_closed(s, R2):
    """Ricci curvatures (fibre, base, base) of the Berger sphere of squashing s and base scale R_2:
    (2 s^2, 4 - 2 s^2, 4 - 2 s^2) / R_2^2.  Unit S^3 at s = R_2 = 1."""
    s = np.asarray(s, dtype=float)
    R2 = np.asarray(R2, dtype=float)
    return np.stack([2 * s * s, 4 - 2 * s * s, 4 - 2 * s * s], axis=-1) / (R2 ** 2)[..., None]


def slice_ricci_along(sol: Dict[str, np.ndarray]) -> np.ndarray:
    """Ricci curvatures of the slice at every t > 0 of the solution, from qgl_pancake.slice_ricci, shape (n, 3)."""
    pos = sol["t"] > 0
    return np.array([qp.slice_ricci(r) for r in sol["R"][pos]])


def fibre_turns_leading(a0: float, t, t_in: float):
    """Leading-order turns of the fibre by light between t_in and t: (2 / a0) ln(t / t_in) / (4 pi)."""
    return hz.fibre_log_coefficient(a0) * np.log(np.asarray(t) / t_in) / hz.FIBRE_PERIOD


def efolds_per_turn(R1, t):
    """Local e-folds of t per turn of the fibre: d ln t = 4 pi R_1 / (2 t) = 2 pi R_1 / t.  Tends to 2 pi a0 as t -> 0.
    (The Hopf-fibration repository's hopf_berger.levels_per_turn is the same number from the other side.)"""
    return 2.0 * np.pi * np.asarray(R1) / np.asarray(t)


def summary(a0: float = 1.0, b0: float = 3.0, f0: float = 2.0, m: float = 0.5, t_max: float = 6.0) -> Dict[str, float]:
    sol = qp.pancake_solution(a0, b0, f0, m, t_max=t_max)
    b = berger_parameter(sol)
    ric = slice_ricci_along(sol)
    scalar = ric.sum(axis=1)
    h = hz.pancake_horizons(sol)
    i_round = int(np.argmax(b["s"] >= 1.0)) if np.any(b["s"] >= 1.0) else -1
    return dict(s_first=float(b["s"][0]), s_last=float(b["s"][-1]),
                t_round=float(b["t"][i_round]) if i_round >= 0 else float("nan"),
                scalar_first_times_b0sq=float(scalar[0] * b0 ** 2),
                efolds_per_turn_first=float(efolds_per_turn(b["R1"][0], b["t"][0])),
                two_pi_a0=2 * np.pi * a0,
                fibre_turns_last=float(h["fibre_turns"][-1]),
                fibre_turns_leading_last=float(fibre_turns_leading(a0, b["t"][-1], b["t"][0])))


def plot_hopf(path: str, a0: float = 1.0, b0: float = 3.0, f0: float = 2.0, m: float = 0.5) -> None:
    """Three panels: squashing s(t); slice Ricci curvature, true and leading order; fibre turns and e-folds per turn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sol = qp.pancake_solution(a0, b0, f0, m, t_max=6.0, n=800)
    b = berger_parameter(sol)
    t = b["t"]
    ric = slice_ricci_along(sol)
    h = hz.pancake_horizons(sol)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    a = ax[0]
    a.plot(t, b["s"], color=_PAL[1], lw=1.8, label=r"$s=R_1/R_2$ (numerical solution)")
    a.plot(t, a0 * t / b0, color=_PAL[1], lw=1.2, ls="--", label=r"leading order $a_0 t/b_0$")
    a.axhline(1.0, color="0.4", lw=0.9)
    a.text(t[0] * 1.5, 1.12, r"$s=1$: round $S^3$", fontsize=8, color="0.3")
    a.set(xscale="log", yscale="log", xlabel="t after the pancake", ylabel="squashing of the Hopf fibre",
          title="The fibre collapses, the base does not")
    a.legend(frameon=False, fontsize=8, loc="lower right")

    a = ax[1]
    names = ["fibre", "base", "base"]
    for k, (col, ls) in enumerate(zip((_PAL[1], _PAL[0]), ("-", "-"))):
        a.plot(t, ric[:, k] * b0 ** 2, color=col, lw=1.8, ls=ls, label=f"Ricci, {names[k]} direction (solution)")
    clo = berger_ricci_closed(b["s"], b["R2"]) * b0 ** 2
    a.plot(t, clo[:, 0], color="k", lw=1.0, ls=":", label=r"Berger closed form $(2s^2, 4-2s^2)/R_2^2$")
    a.plot(t, clo[:, 1], color="k", lw=1.0, ls=":")
    a.plot(t, ric.sum(axis=1) * b0 ** 2, color=_PAL[2], lw=1.6, label=r"scalar curvature (finite at the pancake, $\to 8/b_0^2$)")
    a.set(xscale="log", xlabel="t after the pancake", ylabel=r"curvature $\times\, b_0^2$",
          title="Slice curvature stays finite")
    a.legend(frameon=False, fontsize=8, loc="center left")

    a = ax[2]
    a.plot(t, h["fibre_turns"], color=_PAL[1], lw=1.8, label=f"turns of a light ray round the fibre (from $t_{{in}}={t[0]:.0e}$)")
    a.plot(t, fibre_turns_leading(a0, t, t[0]), color=_PAL[1], lw=1.1, ls="--", label=r"leading order $(2/a_0)\ln(t/t_{in})/4\pi$")
    a.set(xscale="log", xlabel="t after the pancake", ylabel="turns of the fibre", title="Winding and e-folds per turn")
    a2 = a.twinx()
    a2.plot(t, efolds_per_turn(b["R1"], t), color=_PAL[0], lw=1.6, label=r"e-folds of $t$ per turn, $2\pi R_1/t$")
    a2.axhline(2 * np.pi * a0, color=_PAL[0], lw=0.9, ls=":")
    a2.set_ylabel("e-folds of t per fibre turn")
    l1, n1 = a.get_legend_handles_labels()
    l2, n2 = a2.get_legend_handles_labels()
    a.legend(l1 + l2, n1 + n2, frameon=False, fontsize=8, loc="upper right")
    for q in ax:
        q.grid(alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    s = summary()
    print("Pancake as a Berger sphere (a0=1, b0=3, f0=2, m=0.5):")
    print(f"  s = R1/R2: {s['s_first']:.2e} at the first grid time, {s['s_last']:.3f} at t=6; round (s=1) at t = {s['t_round']:.2f}")
    print(f"  slice scalar curvature x b0^2 at the first grid time: {s['scalar_first_times_b0sq']:.6f}   (8 as s -> 0)")
    print(f"  e-folds of t per fibre turn at the first grid time: {s['efolds_per_turn_first']:.4f}   (2 pi a0 = {s['two_pi_a0']:.4f})")
    print(f"  fibre turns by t=6: {s['fibre_turns_last']:.3f}; leading order {s['fibre_turns_leading_last']:.3f}")


if __name__ == "__main__":
    import os
    import sys
    if "--figure" in sys.argv:
        os.makedirs("figures", exist_ok=True)
        plot_hopf(os.path.join("figures", "qgl_hopf.png"))
    else:
        main()
