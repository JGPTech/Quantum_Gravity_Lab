#!/usr/bin/env python3
"""
qgl_horizon.py -- what a pancake singularity does to particle horizons.

The idea under test: "two directions stay open and extended at the singularity, so cosmological horizons are broken
along those axes, allowing communication across arbitrary cylindrical distances".  Checked here against the
metrics we already have.  Short version of what comes out (the tests pin each statement):

* In the pancake model the axis that collapses is R_1 ~ a0 t, the closed Hopf-fibre direction.  A radial null ray
  along it covers a comoving angle  h_1 = int 2 dt / R_1 ~ (2/a0) ln(t / t_in),  which diverges as t_in -> 0.  So
  the *collapsing* direction has no particle horizon.  But that direction is a circle (period 4 pi in the
  standard Euler-angle S^3; the identification is not checked against the paper) whose proper circumference
  2 pi R_1 -> 0, so "arbitrary distance" there means winding ever more times round a vanishing circle.
* The two *open* directions (R_2 = R_3 -> b0, finite) have a finite comoving horizon, h_2 ~ 2 t / b0: light covers
  a proper distance ~ t, exactly as in an ordinary big bang.  Their horizon is not broken.  Two antipodal points
  of the base S^2 first come into causal contact when  int 2 dt / R_2 = pi, which takes t ~ pi b0 / 2.
* Kasner exponents give the general rule: the comoving horizon along axis i is  t^(1-p_i)/(1-p_i)  for p_i < 1 and
  diverges for p_i >= 1.  Vacuum Kasner has p_i <= 1 with equality only for (1, 0, 0), so the pancake is the
  marginal case: the only vacuum exponent set with a divergent horizon, and then only along one axis.
* The model also continues through t = 0 to t < 0, so "the horizon at the singularity" is partly a statement about
  where one starts counting.  With the continuation the null ray's comoving range is finite on the open axes.

Mixmaster (second half of the module).  Misner's hope was that the endless change of which axis expands makes
horizons vanish in every direction.  With the usual criterion h_i = int dt / a_i (exact comoving range of a null
ray along axis i in the Abelian Bianchi I limit; a proxy for Bianchi IX, which I have not checked against real null
geodesics) the answer here is the opposite: every axis has a FINITE horizon.  Reason: b_i = t / a_i obeys
d ln b_i / d ln t = 1 - v_i, and 1 - v_i = 1 - p_i > 0 inside every vacuum Kasner epoch, so b_i shrinks towards
the singularity epoch after epoch and h_i = int b_i d ln t converges.  The only way round this is p_i = 1 (the
pancake exponents), which Mixmaster never has.  The bounces themselves briefly push v_i to about 1.1 (measured
below), so b_i can grow a little there; that does not change the conclusion in the runs made.
The infinite epoch sequence is handled with a sharp-wall BKL map in multi-precision arithmetic, calibrated against
the numerical Bianchi IX solution for its first epochs (`mixmaster_horizon_ode`).  Misner's own argument concerned the
Hubble horizon of a momentarily static direction, not this integral, and Chitre (1972, Misner's student) found that
the set of initial conditions with light circumnavigating the universe has measure going to zero as the volume
shrinks; I have only read the abstract of that thesis.

Everything here is geometry of our own model; the coupling to any sensor is not involved.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.integrate import cumulative_trapezoid

import qgl_kasner as kn
import qgl_pancake as qp

FIBRE_PERIOD = 4.0 * np.pi      # period of the Euler angle psi on the standard S^3 (identification not checked)


# --------------------------------------------------------------------------
#  Kasner: closed forms
# --------------------------------------------------------------------------

def horizon_type(p: float, tol: float = 1e-12) -> str:
    """'finite' for p < 1, 'logarithmic' for p = 1 (int dt/t), 'power' for p > 1 (not reachable in vacuum)."""
    if abs(p - 1.0) <= tol:
        return "logarithmic"
    return "finite" if p < 1.0 else "power"


def kasner_comoving_horizon(p: float, t: float) -> float:
    """int_0^t t'^(-p) dt' = t^(1-p)/(1-p) for p < 1; inf for p >= 1.  Metric a_i = t^p_i, null ray along axis i."""
    return float(t ** (1.0 - p) / (1.0 - p)) if p < 1.0 - 1e-12 else float("inf")


def kasner_proper_horizon(p: float, t: float) -> float:
    """a_i(t) * comoving horizon = t / (1 - p) for p < 1.  Finite and ~ t on every non-collapsing axis."""
    return float(t / (1.0 - p)) if p < 1.0 - 1e-12 else float("inf")


def kasner_horizon_table(us: Sequence[float] = (1.5, 2.0, 3.0, 5.0)) -> list:
    """Per u: exponents, and which axes have a divergent comoving horizon (none, for vacuum Kasner with u > 1)."""
    rows = []
    for u in us:
        p = kn.kasner_exponents(u)
        rows.append(dict(u=u, p=tuple(float(x) for x in p),
                         types=tuple(horizon_type(float(x)) for x in p)))
    rows.append(dict(u=None, p=(1.0, 0.0, 0.0), types=tuple(horizon_type(x) for x in (1.0, 0.0, 0.0))))
    return rows


# --------------------------------------------------------------------------
#  Pancake: numerical horizons on t > 0
# --------------------------------------------------------------------------

def pancake_horizons(sol: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Comoving null-ray angles for t > 0 along a `pancake_solution`, counted from t_in = the first grid time
    (the solution skips |t| < t_start).

        fibre : int_{t_in}^t 2 dt / R_1          (grows like (2/a0) ln(t/t_in); diverges as t_in -> 0)
        base  : int_0^t   2 dt / R_2             (finite; the piece on [0, t_in] is added as 2 t_in / b0)

    plus the proper size of the fibre circle, 2 pi R_1, and the proper distance light has covered along the
    base, int dt = t (a null ray has |dx_proper/dt| = 1 in a comoving frame, so this is just elapsed time).
    Keys: t, fibre, base, fibre_circumference, proper_reach, fibre_turns.
    """
    pos = sol["t"] > 0
    t = sol["t"][pos]
    R1, R2 = sol["R"][pos, 0], sol["R"][pos, 1]
    fibre = cumulative_trapezoid(2.0 / R1, t, initial=0.0)
    base = cumulative_trapezoid(2.0 / R2, t, initial=0.0) + 2.0 * t[0] / sol["params"]["b0"]
    return dict(t=t, fibre=fibre, base=base, fibre_circumference=2.0 * np.pi * R1,
                proper_reach=t.copy(), fibre_turns=fibre / FIBRE_PERIOD)


def fibre_log_coefficient(a0: float) -> float:
    """Leading growth of the fibre horizon: h_1 = (2/a0) ln(t/t_in).  This is the same 2/a0 as in x_winding."""
    return 2.0 / a0


def base_contact_time(hz: Dict[str, np.ndarray], angle: float = np.pi) -> float:
    """First t at which the base null angle reaches `angle` (pi: antipodal points of the base S^2); nan if not reached."""
    idx = np.flatnonzero(hz["base"] >= angle)
    if len(idx) == 0:
        return float("nan")
    i = idx[0]
    if i == 0:
        return float(hz["t"][0])
    t0, t1, h0, h1 = hz["t"][i - 1], hz["t"][i], hz["base"][i - 1], hz["base"][i]
    return float(t0 + (angle - h0) * (t1 - t0) / (h1 - h0))


def fibre_time_to_wrap(a0: float, t: float, turns: float = 1.0) -> float:
    """Latest t_in such that a ray started at t_in has wound `turns` times by time t:  t_in = t exp(-turns 4 pi a0 / 2)."""
    return float(t * np.exp(-turns * FIBRE_PERIOD * a0 / 2.0))


# --------------------------------------------------------------------------
#  Mixmaster
# --------------------------------------------------------------------------

def mixmaster_horizon_ode(mm, n: int = 400000) -> Dict[str, np.ndarray]:
    """
    h_i(t_cut) = int_{t_cut}^{t0} dt / a_i along a numerical `qgl_bkl.Mixmaster` (t = time to the singularity, so
    t_cut -> 0 is the approach to it), on a log grid of n points.  Keys: t, h (n, 3), v_max (3,).
    """
    lnt = np.linspace(np.log(mm.t0), np.log(mm.t_end), n)
    t = np.exp(lnt)
    x, v = mm.state(t)
    h = cumulative_trapezoid(t[:, None] * np.exp(-x), -lnt, axis=0, initial=0.0)
    return dict(t=t, h=h, v_max=v.max(axis=0))


def epoch_amplitudes(mm, n_grid: int = 200000):
    """Per Kasner epoch of a `Mixmaster` run: (exponents p, amplitudes A with ln a_i = A_i + p_i ln t)."""
    out = []
    for e in mm.epochs(n_grid=n_grid):
        tm = float(np.sqrt(e["t_start"] * e["t_end"]))
        x, _ = mm.state(tm)
        out.append((e["v"], x[0] - e["v"] * np.log(tm)))
    return out


def mixmaster_horizon_map(u0, A0: Sequence[float], ln_t0: float, n_epochs: int, theta: float = 0.5,
                          dps: int = 0) -> Dict[str, list]:
    """
    Horizon integrals through n_epochs Kasner epochs of the sharp-wall BKL map, in multi-precision (mpmath).

    Epoch k has ln a_i = A_i + p_i s (s = ln t, decreasing).  The wall term of the negative axis n is
    (1/2) exp(4 (A_n - sum A / 2) + 4 p_n s); the bounce is placed where it reaches `theta` (0.5 reproduces the numerical
    amplitudes of qgl_bkl to about 0.5 in ln a), exponents follow `qgl_bkl.bkl_transition`, and ln a_i is continuous.
    Each epoch adds  e^(-A_i) [e^(q s_start) - e^(q s_end)] / q,  q = 1 - p_i, to h_i.
    `u0` may be an mpmath number (use an irrational one: rational u terminates the BKL map).  `A0` is the first epoch's
    amplitudes (axis order: negative, middle, large exponent) starting at ln t = `ln_t0`.
    Returns per-epoch lists: p, s_bounce, log10_b (b_i = t/a_i at the epoch start) and log10_h (cumulative).
    """
    import mpmath as mp
    mp.mp.dps = dps or (60 + 25 * n_epochs)
    u = mp.mpf(u0)
    D = 1 + u + u * u
    p = [-u / D, (1 + u) / D, u * (1 + u) / D]
    A = [mp.mpf(a) for a in A0]
    s_start = mp.mpf(ln_t0)
    h = [mp.mpf(0)] * 3
    out = dict(p=[], s_bounce=[], log10_b=[], log10_h=[])
    for _ in range(n_epochs):
        n = min(range(3), key=lambda i: p[i])
        c = 4 * (A[n] - sum(A) / 2)
        sb = (mp.log(2 * theta) - c) / (4 * p[n])
        sb = min(sb, s_start)                       # a bounce cannot lie before the epoch starts
        out["p"].append([float(x) for x in p])
        out["log10_b"].append([float(mp.log10(mp.e ** (s_start - A[i] - p[i] * s_start))) for i in range(3)])
        for i in range(3):
            q = 1 - p[i]
            h[i] += mp.e ** (-A[i] + q * s_start) * (1 - mp.e ** (q * (sb - s_start))) / q
        out["s_bounce"].append(float(sb))
        out["log10_h"].append([float(mp.log10(x)) if x > 0 else float("-inf") for x in h])
        den = 1 + 2 * p[n]
        pn = [(p[i] + 2 * p[n]) / den for i in range(3)]
        pn[n] = abs(p[n]) / den
        A = [A[i] + (p[i] - pn[i]) * sb for i in range(3)]
        p, s_start = pn, sb
    return out


def summary(a0: float = 1.0, b0: float = 3.0, f0: float = 2.0, m: float = 0.5, t_max: float = 6.0) -> Dict[str, float]:
    sol = qp.pancake_solution(a0, b0, f0, m, t_max=t_max)
    hz = pancake_horizons(sol)
    i1 = int(np.searchsorted(hz["t"], 1.0))
    return dict(t_contact_base=base_contact_time(hz),
                pi_b0_over_2=float(np.pi * b0 / 2.0),
                fibre_turns_t1=float(hz["fibre_turns"][i1]),
                t_in=float(hz["t"][0]),
                fibre_circumference_t1=float(hz["fibre_circumference"][i1]),
                base_angle_t1=float(hz["base"][i1]))


# --------------------------------------------------------------------------
#  Figures
# --------------------------------------------------------------------------

_PAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]   # Okabe-Ito


def plot_pancake_horizons(path: str, a0: float = 1.0, b0: float = 3.0, f0: float = 2.0, m: float = 0.5) -> None:
    """Three panels: null-ray angles round the fibre and across the base; proper sizes; Kasner horizon vs exponent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sol = qp.pancake_solution(a0, b0, f0, m, t_max=6.0)
    hz = pancake_horizons(sol)
    t, tc = hz["t"], base_contact_time(pancake_horizons(sol))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    a = ax[0]
    a.plot(t, hz["fibre"], color=_PAL[1], lw=1.8, label=r"fibre (collapsing, $R_1\approx a_0 t$), from $t_{in}=10^{-4}$")
    t_in = 1e-9
    tt = np.geomspace(t_in, 1.0, 200)
    a.plot(tt, fibre_log_coefficient(a0) * np.log(tt / t_in), color=_PAL[1], lw=1.2, ls="--",
           label=r"$(2/a_0)\ln(t/t_{in})$, $t_{in}=10^{-9}$")
    a.plot(t, hz["base"], color=_PAL[0], lw=1.8, label=r"base $S^2$ (open, $R_2\to b_0$)")
    a.axhline(np.pi, color="0.5", lw=0.9)
    a.text(3e-9, np.pi * 0.62, r"$\pi$: antipodal points of the base", fontsize=8, color="0.3")
    a.axhline(FIBRE_PERIOD, color="0.7", lw=0.9, ls=":")
    a.text(3e-9, FIBRE_PERIOD * 1.08, r"$4\pi$: one turn of the fibre", fontsize=8, color="0.4")
    a.axvline(tc, color=_PAL[0], lw=0.9, ls=":")
    a.set(xscale="log", yscale="log", ylim=(1e-3, 80), xlim=(1e-9, 12), xlabel="t after the pancake", ylabel="comoving null-ray angle",
          title="Which direction has no horizon?")
    a.legend(frameon=False, fontsize=8, loc="lower right")

    a = ax[1]
    a.plot(t, hz["fibre_circumference"], color=_PAL[1], lw=1.8, label=r"fibre circumference $2\pi R_1$")
    a.plot(t, 2 * np.pi * sol["R"][sol["t"] > 0][:, 1], color=_PAL[0], lw=1.8, label=r"base circumference $2\pi R_2$")
    a.plot(t, hz["proper_reach"], color="0.2", lw=1.4, ls="--", label="distance light has covered ($=t$)")
    a.set(xscale="log", yscale="log", xlabel="t after the pancake", ylabel="proper length",
          title="Open directions stay big, light reaches only about t")
    a.legend(frameon=False, fontsize=8, loc="lower right")

    a = ax[2]
    p = np.linspace(-0.5, 0.995, 400)
    a.plot(p, [kasner_comoving_horizon(x, 1.0) for x in p], color="0.25", lw=1.6)
    for u, col in zip((1.5, 2.0, 3.0, 5.0), _PAL[2:6]):
        for x in kn.kasner_exponents(u):
            a.plot([float(x)], [kasner_comoving_horizon(float(x), 1.0)], "o", color=col, ms=5,
                   label=f"vacuum Kasner u={u}" if x == kn.kasner_exponents(u)[0] else None)
    a.axvline(1.0, color=_PAL[1], lw=1.2, ls="--")
    a.text(-0.45, 5.6, r"$p=1$: pancake," + "\nlogarithmic divergence", color=_PAL[1], fontsize=8)
    a.set(ylim=(0, 8), xlabel=r"Kasner exponent $p_i$", ylabel=r"comoving horizon $\int_0^1 t^{-p}dt$",
          title=r"Only $p\geq 1$ breaks the horizon")
    a.legend(frameon=False, fontsize=8, loc="upper left")
    for q in ax:
        q.grid(alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mixmaster_horizons(path: str, n_epochs: int = 40) -> None:
    """Three panels: ODE horizon vs approach to the singularity; map horizon vs epoch; t/a_i at epoch starts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mpmath as mp
    import qgl_bkl as bk
    u = float(np.pi)
    mm = bk.Mixmaster(u, bk.clean_t0(u), 1.0, 1e-40)
    ode = mixmaster_horizon_ode(mm, n=200000)
    (_, A1) = epoch_amplitudes(mm)[0]
    mp.mp.dps = 60 + 25 * n_epochs
    runs = {th: mixmaster_horizon_map(mp.pi, A1, float(np.log(mm.t0)), n_epochs, theta=th)
            for th in (0.1, 0.5, 5.0)}
    cols = _PAL[:3]
    names = ["axis 1", "axis 2", "axis 3"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    a = ax[0]
    for i in range(3):
        a.plot(np.log10(ode["t"]), np.log10(np.maximum(ode["h"][:, i], 1e-300)), color=cols[i], lw=1.8, label=names[i])
    a.set(xlabel=r"$\log_{10}$ (time to singularity)", ylabel=r"$\log_{10}\int_{t_{cut}}^{t_0} dt/a_i$",
          ylim=(-9, 0), xlim=(8, -41), title="Numerical Bianchi IX: horizon stops growing")
    a.legend(frameon=False, fontsize=8, loc="center right")

    a = ax[1]
    k = np.arange(1, n_epochs + 1)
    for th, ls in zip(runs, (":", "-", "--")):
        h = np.array(runs[th]["log10_h"])
        for i in range(3):
            a.plot(k, h[:, i], color=cols[i], lw=1.5, ls=ls, label=names[i] if th == 0.5 else None)
    a.plot([], [], color="0.3", ls=":", label=r"wall $\theta=0.1$")
    a.plot([], [], color="0.3", ls="-", label=r"$\theta=0.5$")
    a.plot([], [], color="0.3", ls="--", label=r"$\theta=5$")
    a.set(xlabel="Kasner epoch (BKL map, $u_0=\\pi$)", ylabel=r"$\log_{10}h_i$ (cumulative)",
          title="Infinitely many epochs: the sum converges")
    a.legend(frameon=False, fontsize=8, ncol=2, loc="center right")

    a = ax[2]
    b = np.array(runs[0.5]["log10_b"])
    for i in range(3):
        a.plot(k, -b[:, i], color=cols[i], lw=1.6, label=names[i])
    a.set(yscale="log", xlabel="Kasner epoch", ylabel=r"$-\log_{10}(t/a_i)$ at the epoch start",
          title=r"$t/a_i$ shrinks in every epoch ($1-p_i>0$)")
    a.legend(frameon=False, fontsize=8)
    for q in ax:
        q.grid(alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(directory: str = "figures") -> None:
    import os
    os.makedirs(directory, exist_ok=True)
    plot_pancake_horizons(os.path.join(directory, "qgl_horizon_pancake.png"))
    plot_mixmaster_horizons(os.path.join(directory, "qgl_horizon_mixmaster.png"))


def mixmaster_report() -> None:
    import mpmath as mp
    import qgl_bkl as bk
    u = float(np.pi)
    mm = bk.Mixmaster(u, bk.clean_t0(u), 1.0, 1e-40)
    ode = mixmaster_horizon_ode(mm, n=200000)
    (p1, A1) = epoch_amplitudes(mm)[0]
    mp.mp.dps = 200
    mp_run = mixmaster_horizon_map(mp.pi, A1, float(np.log(mm.t0)), 40)
    n_ep = len(mm.epochs())
    print("\nMixmaster, u0=pi numerical solution to t=1e-40: h_i =", np.array2string(ode["h"][-1], precision=3),
          " max v_i =", np.round(ode["v_max"], 3))
    print(f"  log10 of those: {np.round(np.log10(ode['h'][-1]), 3)}  ({n_ep} epochs)")
    print("BKL sharp-wall map, u0=pi, 40 epochs: log10 h_i after epochs 1, 3, 10, 40:")
    for k in (0, 2, 9, 39):
        print(f"  epoch {k + 1:2d}: log10 h = {np.round(mp_run['log10_h'][k], 3)}")


def main() -> None:
    print("Kasner comoving horizon type per axis (t^(1-p)/(1-p) for p<1, divergent for p>=1)")
    for r in kasner_horizon_table():
        label = "pancake" if r["u"] is None else f"u={r['u']}"
        print(f"  {label:8s} p=({r['p'][0]:+.3f}, {r['p'][1]:+.3f}, {r['p'][2]:+.3f})  {r['types']}")
    s = summary()
    print("\nPancake (a0=1, b0=3, f0=2, m=0.5), light from the pancake outward:")
    print(f"  base S^2 antipodal contact at t = {s['t_contact_base']:.2f}   (pi b0/2 = {s['pi_b0_over_2']:.2f} if R_2 stayed b0)")
    print(f"  fibre turns by t=1, counted from t_in={s['t_in']:.0e}: {s['fibre_turns_t1']:.2f}  (grows like ln(1/t_in))")
    print(f"  fibre circumference at t=1: {s['fibre_circumference_t1']:.2f};  base angle covered by t=1: {s['base_angle_t1']:.2f} rad of pi")
    mixmaster_report()


if __name__ == "__main__":
    import sys
    if "--figures" in sys.argv:
        make_figures()
    else:
        main()
