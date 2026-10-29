#!/usr/bin/env python3
"""
qgl_tokamak.py -- a stylised tokamak mode for the Quantum Gravity Lab.

Replaces the Keplerian orbit of the main QGL pipeline by a charged ion packet
moving in an axisymmetric tokamak field, keeps the two QGL ingredients that
make sense there, and says plainly which parts are physics and which are
stylisation.

PHYSICS (checked in tests/test_tokamak.py)
  * Field.  Large-aspect-ratio toroidal field B_phi = B0 R0 / R plus a poloidal
    field derived from a flux function psi(R, Z) = B0 rho^2 / (2 q0),
    rho^2 = (R - R0)^2 + Z^2, as B_pol = grad(psi) x phi_hat / R.  Deriving it
    from psi makes div B = 0 exactly and gives an exact invariant for the
    unperturbed motion: P_phi = m R v_phi + q psi.
  * Orbit.  Full Lorentz-force motion (not guiding-centre) in Cartesian
    coordinates, units B0 = 1, q/m = 1 (gyrofrequency 1), R0 = 10 by default.
  * Invariants.  Energy and P_phi are conserved for the unperturbed field, and
    the magnetic moment mu = m v_perp^2 / 2B is conserved to the usual
    adiabatic accuracy.
  * Trapped / passing.  Launched at the outer midplane, an ion is magnetically
    trapped when (v_par/v)^2 < 1 - B_min/B_max = 2 rho0 / (R0 + rho0); the
    code classifies orbits by counting reversals of v_par and the tests compare
    with that threshold away from the boundary.

STYLISED (no claim of fusion relevance)
  * "Complexity" is used as a turbulence proxy: a radial profile
    C_T(rho) = 1 / (1 + (rho / rho_t)^p), the same functional form as the QGL
    complexity C_Q(r), multiplies the amplitude of an optional electrostatic
    wave  dphi = A C_T(rho) cos(m theta - n phi - w t).  A = 0 recovers the
    exactly integrable case.  This is a toy perturbation, not a turbulence model.
  * The Jaynes-Cummings sensor of the main QGL core is driven by the local
    field strength (atom frequency omega_a = omega_a0 |B|/B0) so that a trapped
    ion's bounce motion is imprinted on it; see `run_sensor`.

Run `python qgl_tokamak.py` for a short demonstration.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from unified_sandbox_core import Config, JCComplexityManager
from qgl_jc_lindblad import LindbladJCEngine


# --------------------------------------------------------------------------
#  configuration
# --------------------------------------------------------------------------
@dataclass
class TokamakConfig:
    R0: float = 10.0           # major radius
    a: float = 3.0             # minor radius (wall at rho = a)
    B0: float = 1.0            # field on axis
    q0: float = 2.0            # safety factor (constant in this model)
    v: float = 0.05            # ion speed, so the Larmor radius is v / B0
    # stylised perturbation: dphi = A * C_T(rho) * cos(m theta - n phi - w t)
    A: float = 0.0
    m: int = 3
    n: int = 1
    w: float = 0.02
    rho_t: float = 1.5         # turbulence-proxy transition radius
    p_t: float = 2.0           # turbulence-proxy exponent
    # sensor
    jc_dt_per_time: float = 0.02   # JC time units per unit of orbit time

    @property
    def eps(self) -> float:
        return self.a / self.R0


def turbulence_proxy(rho, cfg: TokamakConfig):
    """C_T(rho) = 1 / (1 + (rho/rho_t)^p): same form as the QGL complexity C_Q."""
    return 1.0 / (1.0 + (np.asarray(rho) / cfg.rho_t) ** cfg.p_t)


# --------------------------------------------------------------------------
#  field
# --------------------------------------------------------------------------
def psi(R, Z, cfg: TokamakConfig):
    return cfg.B0 * ((R - cfg.R0) ** 2 + Z ** 2) / (2.0 * cfg.q0)


def B_field(x, y, z, cfg: TokamakConfig):
    """Cartesian B = B_pol + B_phi phi_hat at (x, y, z); z is the vertical axis."""
    R = np.hypot(x, y)
    cph, sph = x / R, y / R
    BR = -cfg.B0 * z / (cfg.q0 * R)
    Bz = cfg.B0 * (R - cfg.R0) / (cfg.q0 * R)
    Bphi = cfg.B0 * cfg.R0 / R
    return np.array([BR * cph - Bphi * sph, BR * sph + Bphi * cph, Bz])


def E_field(x, y, z, t, cfg: TokamakConfig):
    """E = -grad(dphi) for dphi = A C_T(rho) cos(m theta - n phi - w t).

    The gradient is taken numerically (central differences), which keeps the
    code short and is accurate to ~1e-9 relative at the step used."""
    if cfg.A == 0.0:
        return np.zeros(3)

    def dphi(p):
        R = np.hypot(p[0], p[1])
        rho = np.hypot(R - cfg.R0, p[2])
        th = np.arctan2(p[2], R - cfg.R0)
        ph = np.arctan2(p[1], p[0])
        return cfg.A * turbulence_proxy(rho, cfg) * np.cos(cfg.m * th - cfg.n * ph - cfg.w * t)

    h = 1e-5
    g = np.empty(3)
    p = np.array([x, y, z], dtype=float)
    for i in range(3):
        d = np.zeros(3)
        d[i] = h
        g[i] = (dphi(p + d) - dphi(p - d)) / (2 * h)
    return -g


def rhs(t, s, cfg: TokamakConfig):
    x, y, z, vx, vy, vz = s
    B = B_field(x, y, z, cfg)
    E = E_field(x, y, z, t, cfg)
    v = np.array([vx, vy, vz])
    a = E + np.cross(v, B)                       # q/m = 1
    return [vx, vy, vz, a[0], a[1], a[2]]


# --------------------------------------------------------------------------
#  orbit
# --------------------------------------------------------------------------
def initial_state(cfg: TokamakConfig, rho0: float, pitch: float, gyrophase: float = 0.0):
    """Ion at the outer midplane, radius rho0 from the axis, speed cfg.v.

    `pitch` = v_parallel / v (parallel = along B).  The velocity is built from
    the local field direction so that pitch is exact at t = 0."""
    x0 = np.array([cfg.R0 + rho0, 0.0, 0.0])
    B = B_field(*x0, cfg)
    b = B / np.linalg.norm(B)
    e1 = np.cross(b, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(b, e1)
    perp = np.cos(gyrophase) * e1 + np.sin(gyrophase) * e2
    vel = cfg.v * (pitch * b + np.sqrt(max(1 - pitch ** 2, 0.0)) * perp)
    return np.concatenate([x0, vel])


def integrate(cfg: TokamakConfig, s0, t_max: float, n_out: int = 4000,
              rtol: float = 1e-10, atol: float = 1e-12):
    """Integrate until t_max or until the ion hits the wall (rho > a)."""
    def wall(t, s, cfg):
        R = np.hypot(s[0], s[1])
        return cfg.a - np.hypot(R - cfg.R0, s[2])
    wall.terminal = True
    wall.direction = -1.0
    t_eval = np.linspace(0.0, t_max, n_out)
    sol = solve_ivp(rhs, [0.0, t_max], s0, args=(cfg,), t_eval=t_eval, events=wall,
                    method="DOP853", rtol=rtol, atol=atol)
    return sol


def diagnostics(cfg: TokamakConfig, sol):
    """Time series and invariants along an integrated orbit."""
    x, y, z, vx, vy, vz = sol.y
    R = np.hypot(x, y)
    rho = np.hypot(R - cfg.R0, z)
    theta = np.arctan2(z, R - cfg.R0)
    vphi = (-y * vx + x * vy) / R
    Bv = np.array([B_field(x[i], y[i], z[i], cfg) for i in range(len(x))]).T
    Bmag = np.linalg.norm(Bv, axis=0)
    v2 = vx ** 2 + vy ** 2 + vz ** 2
    vpar = (Bv[0] * vx + Bv[1] * vy + Bv[2] * vz) / Bmag
    mu = (v2 - vpar ** 2) / (2.0 * Bmag)
    return dict(t=sol.t, R=R, rho=rho, theta=theta, B=Bmag, vpar=vpar, mu=mu,
                energy=0.5 * v2, Pphi=R * vphi + psi(R, z, cfg),
                lost=bool(sol.status == 1))


def bounce_count(vpar, min_amp: float = 0.0) -> int:
    """Number of sign reversals of v_par, a trapped ion's turning points."""
    s = np.sign(vpar)
    s = s[s != 0]
    return int(np.sum(s[1:] != s[:-1]))


def trapped_threshold_pitch(cfg: TokamakConfig, rho0: float) -> float:
    """|v_par/v| below which an outer-midplane launch is magnetically trapped."""
    return float(np.sqrt(2.0 * rho0 / (cfg.R0 + rho0)))


def theta_winding(theta) -> float:
    """Net poloidal turns (unwrapped theta / 2 pi)."""
    return float((np.unwrap(theta)[-1] - np.unwrap(theta)[0]) / (2 * np.pi))


# --------------------------------------------------------------------------
#  JC sensor
# --------------------------------------------------------------------------
def run_sensor(cfg: TokamakConfig, diag, qgl_cfg: Config | None = None, stride: int = 1):
    """Drive the QGL Jaynes-Cummings sensor along the orbit.

    omega_a = omega_a0 |B|/B0 (Zeeman-like), cavity frequency and coupling
    fixed.  Returns <sigma_z>, <E_Q>, mean photon number and the QGL structural
    complexity C_Q of the sensor.  Uses `LindbladJCEngine`, because the core
    engine's decay step does not keep the density matrix positive.

    Be clear about what this is: the sensor's energy <E_Q> contains omega_a
    explicitly, so it follows |B| along the orbit by construction, on top of the
    sensor's own relaxation (the initial photon population decays, a slow trend
    that has to be removed before the two can be correlated: detrended
    correlation about -0.93 in the demonstration run).  It is a stylised B-meter
    that shows the bounce motion, not an independent measurement of anything."""
    q = qgl_cfg or Config()
    eng, cx = LindbladJCEngine(q), JCComplexityManager(q)
    t, B = diag["t"][::stride], diag["B"][::stride]
    sz, nph, cq, eq = [], [], [], []
    prev = t[0]
    for ti, Bi in zip(t, B):
        params = dict(omega_c=q.omega_cavity_0, omega_a=q.omega_atom_0 * Bi / cfg.B0,
                      g=q.g_0, kappa=q.kappa_0, gamma=q.gamma_0)
        eng.evolve(params, max(ti - prev, 0.0) * cfg.jc_dt_per_time, 1.0)
        prev = ti
        ex = eng.compute_expectations(params)
        st = cx.update(ex, params)
        sz.append(ex["sigma_z"])
        eq.append(ex["E_Q"])
        nph.append(ex["n"])
        cq.append(st["C_Q"])
    return dict(t=t, sigma_z=np.array(sz), n=np.array(nph), C_Q=np.array(cq), E_Q=np.array(eq), B=np.asarray(B))


# --------------------------------------------------------------------------
#  confinement scan (stylised control lever)
# --------------------------------------------------------------------------
def confinement_scan(cfg: TokamakConfig, amplitudes, rho0=1.0, pitch=0.5, t_max=6000.0,
                     n_ions=8, seed=0):
    """Fraction of ions lost to the wall by t_max, for each perturbation amplitude.

    The ions differ in gyrophase only (seeded), so the scan is deterministic."""
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, n_ions)
    out = []
    for A in amplitudes:
        c = TokamakConfig(**{**cfg.__dict__, "A": float(A)})
        lost, dP = 0, []
        for ph in phases:
            sol = integrate(c, initial_state(c, rho0, pitch, ph), t_max, n_out=400, rtol=1e-8, atol=1e-10)
            d = diagnostics(c, sol)
            lost += int(d["lost"])
            dP.append(np.ptp(d["Pphi"]))
        out.append(dict(A=float(A), lost_fraction=lost / n_ions, Pphi_spread=float(np.mean(dP))))
    return out


# --------------------------------------------------------------------------
#  figure
# --------------------------------------------------------------------------
def figure(path="figures/qgl_tokamak.png"):
    """Four panels: poloidal orbits, invariants, perturbation scan, sensor."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = TokamakConfig(q0=1.0)
    rho0 = 2.0
    thr = trapped_threshold_pitch(cfg, rho0)
    runs = {}
    for name, pitch in (("trapped", 0.2), ("passing", 1.25 * thr)):
        sol = integrate(cfg, initial_state(cfg, rho0, pitch), 9000.0, n_out=3000, rtol=1e-9, atol=1e-11)
        runs[name] = diagnostics(cfg, sol)
    amps = [0.0, 1e-3, 3e-3, 1e-2]
    scan = confinement_scan(cfg, amps, rho0=rho0, pitch=0.2, t_max=1500.0, n_ions=3)
    sen = run_sensor(cfg, runs["trapped"], stride=2)

    fig, ax = plt.subplots(2, 2, figsize=(10, 8.6), gridspec_kw=dict(hspace=0.35, wspace=0.3))
    a = ax[0, 0]
    for r in (1, 2, cfg.a):
        th = np.linspace(0, 2 * np.pi, 200)
        a.plot(cfg.R0 + r * np.cos(th), r * np.sin(th), color="0.8", lw=0.8)
    for name, col in (("trapped", "C3"), ("passing", "C0")):
        d = runs[name]
        a.plot(d["R"], d["rho"] * np.sin(d["theta"]), color=col, lw=0.8, label=name)
    a.set_aspect("equal"); a.set_xlabel("R"); a.set_ylabel("Z")
    a.set_title("(a) poloidal projection, launch at outer midplane"); a.legend(frameon=False)
    a = ax[0, 1]
    for name, col in (("trapped", "C3"), ("passing", "C0")):
        d = runs[name]
        a.semilogy(d["t"], np.abs(d["energy"] / d["energy"][0] - 1) + 1e-17, color=col, lw=0.8, label="|dE/E| " + name)
        a.semilogy(d["t"], np.abs(d["Pphi"] - d["Pphi"][0]) + 1e-17, color=col, lw=0.8, ls="--", label="|dP_phi| " + name)
    a.set_xlabel("t"); a.set_title("(b) invariants (A = 0)"); a.legend(frameon=False, fontsize=7)
    a = ax[1, 0]
    a.plot([s["A"] for s in scan], [s["Pphi_spread"] for s in scan], "o-")
    a.set_xlabel("perturbation amplitude A"); a.set_ylabel("mean range of P_phi")
    a.set_title("(c) P_phi is no longer conserved when A > 0")
    a = ax[1, 1]
    d = runs["trapped"]
    a.plot(d["t"], d["B"], color="0.4", lw=0.8, label="|B| along orbit")
    a.set_ylabel("|B|"); a.set_xlabel("t")
    b = a.twinx()
    b.plot(sen["t"], sen["E_Q"], color="C1", lw=0.8, label="sensor <E_Q>"); b.set_ylabel("<E_Q>")
    a.set_title("(d) sensor <E_Q> = relaxation trend + |B| signal")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return scan


if __name__ == "__main__":
    import sys
    if "--figure" in sys.argv:
        print(figure())
        sys.exit(0)
    cfg = TokamakConfig()
    rho0 = 1.0
    print("trapped threshold |pitch| <", round(trapped_threshold_pitch(cfg, rho0), 3))
    for pitch in (0.2, 0.9):
        sol = integrate(cfg, initial_state(cfg, rho0, pitch), 6000.0)
        d = diagnostics(cfg, sol)
        print("pitch %.1f  reversals %d  poloidal turns %.2f  dE/E %.1e  dP_phi %.1e  mu spread %.1e"
              % (pitch, bounce_count(d["vpar"]), theta_winding(d["theta"]),
                 np.ptp(d["energy"]) / d["energy"][0], np.ptp(d["Pphi"]),
                 np.ptp(d["mu"]) / d["mu"].mean()))
