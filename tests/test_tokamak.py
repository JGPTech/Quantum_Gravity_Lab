"""Tests for the tokamak mode (qgl_tokamak.py) and the Lindblad JC engine."""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from unified_sandbox_core import Config, JCQuantumEngine
from qgl_jc_lindblad import LindbladJCEngine
import qgl_tokamak as tk

CFG = tk.TokamakConfig(q0=1.0)
RHO0 = 2.0
THR = tk.trapped_threshold_pitch(CFG, RHO0)


def run(pitch, t_max, cfg=CFG, n_out=1500):
    sol = tk.integrate(cfg, tk.initial_state(cfg, RHO0, pitch), t_max, n_out=n_out, rtol=1e-9, atol=1e-11)
    return tk.diagnostics(cfg, sol)


def test_field_is_divergence_free():
    h = 1e-5
    for p in ([10.5, 0.3, 0.2], [9.0, 1.0, -1.5], [11.0, -2.0, 0.7]):
        div = 0.0
        for i in range(3):
            d = np.zeros(3)
            d[i] = h
            div += (tk.B_field(*(np.array(p) + d), CFG)[i] - tk.B_field(*(np.array(p) - d), CFG)[i]) / (2 * h)
        assert abs(div) < 1e-8


def test_safety_factor_matches_model():
    # q = rho B_phi / (R B_pol) = q0 R0 / R for this psi (equals q0 on the axis)
    R = CFG.R0 + 1.5
    B = tk.B_field(R, 0.0, 0.0, CFG)
    q = 1.5 * abs(B[1]) / (R * abs(B[2]))
    assert q == pytest.approx(CFG.q0 * CFG.R0 / R, rel=1e-9)


def test_invariants_without_perturbation():
    d = run(0.2, 2500.0)
    assert np.ptp(d["energy"]) / d["energy"][0] < 1e-6
    assert np.ptp(d["Pphi"]) < 1e-8            # exact invariant, integrator-limited
    assert np.ptp(d["mu"]) / d["mu"].mean() < 0.1   # adiabatic invariant only


def test_trapped_vs_passing_follow_threshold():
    trapped = run(0.8 * THR, 5000.0)
    passing = run(1.25 * THR, 5000.0)
    assert tk.bounce_count(trapped["vpar"]) >= 2
    assert tk.bounce_count(passing["vpar"]) == 0
    assert abs(tk.theta_winding(trapped["theta"])) < 0.5
    assert abs(tk.theta_winding(passing["theta"])) > 1.0


def test_passing_transit_time():
    pitch = 0.9
    d = run(pitch, 5000.0)
    expected = pitch * CFG.v * 5000.0 / (2 * np.pi * CFG.q0 * CFG.R0)   # poloidal turns
    assert tk.theta_winding(d["theta"]) == pytest.approx(expected, rel=0.15)


def test_deeply_trapped_bounce_period():
    d = run(0.2, 9000.0, n_out=9000)
    s = np.sign(d["vpar"])
    t_rev = d["t"][np.where(s[1:] != s[:-1])[0]]
    assert len(t_rev) >= 3
    half = np.mean(np.diff(t_rev))
    eps = RHO0 / CFG.R0
    theory = 2 * np.pi / (np.sqrt(eps / 2) * CFG.v / (CFG.q0 * CFG.R0))
    assert 2 * half == pytest.approx(theory, rel=0.2)    # observed ~13% above the small-eps formula


def test_perturbation_breaks_pphi_monotonically():
    spread = []
    for A in (0.0, 2e-3, 1e-2):
        c = tk.TokamakConfig(**{**CFG.__dict__, "A": A})
        d = run(0.2, 1200.0, cfg=c, n_out=300)
        spread.append(np.ptp(d["Pphi"]))
    assert spread[0] < 1e-8 < spread[1] < spread[2]


def test_turbulence_proxy_shape():
    r = np.linspace(0, 5, 50)
    c = tk.turbulence_proxy(r, CFG)
    assert c[0] == 1.0 and np.all(np.diff(c) < 0)
    assert tk.turbulence_proxy(CFG.rho_t, CFG) == pytest.approx(0.5)


def test_confinement_scan_is_deterministic():
    a = tk.confinement_scan(CFG, [0.0, 5e-3], rho0=RHO0, pitch=0.2, t_max=300.0, n_ions=2, seed=3)
    b = tk.confinement_scan(CFG, [0.0, 5e-3], rho0=RHO0, pitch=0.2, t_max=300.0, n_ions=2, seed=3)
    assert a == b


def _min_eig(engine):
    return np.linalg.eigvalsh((engine.rho + engine.rho.conj().T) / 2).min()


def _evolve(cls):
    q = Config()
    p = dict(omega_c=q.omega_cavity_0, omega_a=q.omega_atom_0, g=q.g_0, kappa=q.kappa_0, gamma=q.gamma_0)
    e = cls(q)
    for _ in range(200):
        e.evolve(p, 0.1, 1.0)
    return e, p


def test_lindblad_engine_is_positive_and_bounded():
    e, p = _evolve(LindbladJCEngine)
    assert _min_eig(e) > -1e-12
    assert abs(np.trace(e.rho) - 1) < 1e-12
    assert -1.0 <= e.compute_expectations(p)["sigma_z"] <= 1.0


def test_core_decay_breaks_positivity():
    """Documents a bug in unified_sandbox_core.JCQuantumEngine._apply_decays."""
    e, _ = _evolve(JCQuantumEngine)
    assert _min_eig(e) < -1e-3


def test_sensor_follows_B():
    d = run(0.2, 9000.0, n_out=900)
    sen = tk.run_sensor(CFG, d)
    t = sen["t"]
    res = lambda x: x - np.polyval(np.polyfit(t, x, 2), t)    # remove the sensor's own relaxation
    corr = np.corrcoef(res(sen["E_Q"]), res(sen["B"]))[0, 1]
    assert corr < -0.8                     # E_Q ~ -omega_a, so anticorrelated with |B|
    assert np.all(sen["sigma_z"] >= -1.0 - 1e-9)
