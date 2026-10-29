"""Curvature-driven complexity scalar and sensor coupling (qgl_curvature).

Skipped when EinsteinPy is not installed.  All tests are fast.
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

pytest.importorskip("einsteinpy")
sp = pytest.importorskip("sympy")

import qgl_curvature as qc
import qgl_kasner as qk
from qgl_precession import configure_orbit
from unified_sandbox_core import Config, compute_quantum_clock_rate, jc_params_from_classical

K_MOM = (0.3, -0.2, 0.5)
P1 = qk.kasner_exponents(1.0)


def orbit_cfg(complexity_power=0.7):
    cfg = configure_orbit(Config(), 100.0, 0.6)
    cfg.complexity_power = complexity_power
    return cfg


# ---------------------------------------------------------------- Schwarzschild: K reproduces the r-forms

def test_symbolic_schwarzschild_kretschmann_is_48_over_r6():
    t, r, th, ph = sp.symbols("t r theta phi", positive=True)
    f = 1 - 2 / r                                        # G = M = c = 1
    K = qc.kretschmann_of_diagonal_metric([-f, 1 / f, r ** 2, r ** 2 * sp.sin(th) ** 2], (t, r, th, ph))
    assert sp.simplify(K - 48 / r ** 6) == 0


@pytest.mark.parametrize("power", [0.7, 1.5, 2.3])
def test_curvature_complexity_equals_core_C_Q_on_schwarzschild(power):
    cfg = orbit_cfg(power)
    r = np.geomspace(0.05 * cfg.r_q_transition, 20 * cfg.r_q_transition, 40)
    K = qc.schwarzschild_kretschmann(r, cfg)
    K_trans = qc.schwarzschild_kretschmann(cfg.r_q_transition, cfg)
    core = np.array([compute_quantum_clock_rate(x, cfg)[0] for x in r])
    assert qc.complexity_from_curvature(K, K_trans, power) == pytest.approx(core, rel=1e-12)


@pytest.mark.parametrize("r,v", [(30.0, 0.05), (60.0, 0.2), (160.0, 0.4)])
def test_curvature_sensor_coupling_equals_core_on_schwarzschild(r, v):
    cfg = orbit_cfg()
    r_peri = cfg.p / (1.0 + cfg.e)
    phi_hat = (-cfg.G * cfg.M / r) / abs(cfg.jc_phi_ref)
    got = qc.jc_params_from_curvature(float(qc.schwarzschild_kretschmann(r, cfg)), v, cfg,
                                      K_ref=float(qc.schwarzschild_kretschmann(r_peri, cfg)),
                                      v_ref=cfg.jc_v_ref, phi_hat=phi_hat)
    want = jc_params_from_classical(r, v, cfg)
    for key in want:                                     # equal up to the core's 1e-6 regulator on r^2
        assert got[key] == pytest.approx(want[key], rel=1e-6)


# ---------------------------------------------------------------- complexity limits

def test_complexity_limits():
    assert qc.complexity_from_curvature(0.0, 1.0, 1.0) == 0.0            # flat
    assert qc.complexity_from_curvature(1e12, 1.0, 6.0) == pytest.approx(1.0, abs=1e-6)   # (K_trans/K)^(p/6) -> 0
    assert qc.complexity_from_curvature(1e12, 1.0, 1.0) == pytest.approx(1.0 / (1.0 + 1e-2))  # exponent p/6 = 1/6
    assert qc.complexity_from_curvature(2.0, 2.0, 1.7) == pytest.approx(0.5)


def test_density_proxy_is_one_at_reference_and_cube_root():
    assert qc.density_proxy(5.0, 5.0) == pytest.approx(1.0)
    assert qc.density_proxy(8.0, 1.0) == pytest.approx(2.0)


# ---------------------------------------------------------------- sensor on a Kasner worldline

@pytest.fixture(scope="module")
def run():
    return qc.kasner_sensor_run(P1, K_MOM, t0=10.0, t_end=2.0, n_steps=150)


def test_curvature_and_complexity_grow_toward_the_singularity(run):
    for col in ("K", "C_Q", "rho_hat", "omega_c"):
        assert np.all(np.diff(run[col]) > 0.0), col
    assert np.all((run["C_Q"] > 0.0) & (run["C_Q"] < 1.0))


def test_density_proxy_follows_kasner_power_law(run):
    # K ~ t^-4  =>  rho_hat = (K/K_ref)^(1/3) = (t0/t)^(4/3), exactly
    t = run["t"].to_numpy()
    assert run["rho_hat"].to_numpy() == pytest.approx((10.0 / t) ** (4.0 / 3.0), rel=1e-10)


def test_sensor_state_is_physical(run):
    assert np.all(np.abs(run["sigma_z"]) <= 1.0 + 1e-9)
    assert np.all((run["purity"] > 0.0) & (run["purity"] <= 1.0 + 1e-9))
    assert np.all(np.isfinite(run[["n", "sigma_z", "c", "omega_c"]].to_numpy()))


def test_sensor_responds_to_curvature_only_through_s_rho():
    base = Config()
    off = Config()
    off.s_rho = 0.0
    off.s_kappa = 0.0
    a = qc.kasner_sensor_run(P1, K_MOM, 10.0, 2.0, cfg=base, n_steps=150)
    b = qc.kasner_sensor_run(P1, K_MOM, 10.0, 2.0, cfg=off, n_steps=150)
    assert np.allclose(b["omega_c"], base.omega_cavity_0)                 # no curvature dependence
    assert np.max(np.abs(a["omega_c"] - b["omega_c"])) > 1.0              # curvature moves the cavity
    assert np.max(np.abs(a["n"] - b["n"])) > 1e-3                         # and changes what it sees


def test_flat_kasner_has_no_curvature_for_the_sensor_to_read():
    with pytest.raises(ValueError):
        qc.kasner_sensor_run((0.0, 0.0, 1.0), K_MOM, 10.0, 2.0, n_steps=20)
