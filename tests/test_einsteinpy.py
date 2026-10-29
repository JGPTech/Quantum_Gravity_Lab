"""QGL checked against exact general relativity (EinsteinPy and closed forms).

Skipped entirely when EinsteinPy is not installed (pip install einsteinpy).
The tests that integrate EinsteinPy geodesics are marked `slow` and need
`python -m pytest --runslow` (about a minute or two).

Where QGL is only an approximation, the tests pin down *how* it differs
(the size and the scaling with p), not agreement it does not have.
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

pytest.importorskip("einsteinpy")

import qgl_einsteinpy as qe
from unified_sandbox_core import Config, compute_quantum_clock_rate, compute_redshift_factor

slow = pytest.mark.slow

# Orbit used for the geodesic comparisons: p = 60 GM/c^2, e = 0.3 (about 4000 steps).
P, E, SPIN = 60.0, 0.3, 0.9


# ---------------------------------------------------------------- lapse / redshift

@pytest.mark.parametrize("r_geo", [2.5, 3.0, 6.0, 10.0, 100.0, 1.0e4])
def test_clock_rate_matches_exact_schwarzschild_lapse(r_geo):
    cfg = Config()
    r = r_geo * qe.length_unit(cfg)
    qgl = compute_quantum_clock_rate(r, cfg)[2]
    assert qgl == pytest.approx(float(qe.exact_static_lapse(r_geo)), rel=1e-12)


def test_redshift_matches_exact_schwarzschild():
    cfg = Config()
    u = qe.length_unit(cfg)
    exact = float(qe.exact_static_lapse(6.0) / qe.exact_static_lapse(1.0e4))
    assert compute_redshift_factor(6.0 * u, 1.0e4 * u, cfg) == pytest.approx(exact, rel=1e-12)


def test_lapse_rejects_horizon_and_interior():
    with pytest.raises(ValueError):
        qe.exact_static_lapse(2.0)


# ---------------------------------------------------------------- closed-form reference

@pytest.mark.parametrize("p,e", [(60.0, 0.3), (100.0, 0.6), (400.0, 0.1)])
def test_exact_constants_put_turning_points_where_intended(p, e):
    # (du/dphi)^2 = (E^2-1)/L^2 + 2u/L^2 - u^2 + 2u^3 must vanish at u = (1 +/- e)/p
    c = qe.schwarzschild_constants(p, e)
    for u in ((1.0 - e) / p, (1.0 + e) / p):
        f = (c["E"] ** 2 - 1.0) / c["L"] ** 2 + 2.0 * u / c["L"] ** 2 - u ** 2 + 2.0 * u ** 3
        assert abs(f) < 1e-12


@pytest.mark.parametrize("e", [0.3, 0.6])
def test_exact_advance_is_1pn_plus_a_correction_of_order_1_over_p(e):
    # exact / 1PN - 1 is positive, below 6/p, and (times p) converges as p grows
    ratios = {}
    for p in (100.0, 200.0, 400.0, 800.0):
        ratios[p] = qe.schwarzschild_advance_exact(p, e) / (6.0 * np.pi / p) - 1.0
        assert 0.0 < ratios[p] < 6.0 / p
    assert 400.0 * ratios[400.0] == pytest.approx(800.0 * ratios[800.0], rel=0.05)


# ---------------------------------------------------------------- QGL precession vs exact

def test_qgl_calibrated_to_1pn_falls_short_of_exact_by_order_1_over_p():
    """QGL tuned to match 1PN misses the exact advance by a deficit that halves as p doubles."""
    deficit = {}
    for p in (100.0, 200.0, 400.0):
        r = qe.qgl_precession_vs_exact(p, 0.3)
        assert r["qgl_shift"] == pytest.approx(r["one_pn"], rel=1e-4)   # calibration worked
        deficit[p] = 1.0 - r["qgl_over_exact"]
        assert 0.0 < deficit[p] < 6.0 / p
    assert deficit[100.0] / deficit[200.0] == pytest.approx(2.0, rel=0.15)
    assert deficit[200.0] / deficit[400.0] == pytest.approx(2.0, rel=0.15)


# ---------------------------------------------------------------- EinsteinPy geodesics (slow)

@pytest.fixture(scope="module")
def schwarzschild_geodesic():
    return float(qe.einsteinpy_periapsis_advance(P, E)[0])


@pytest.fixture(scope="module")
def kerr_spin_effect():
    """Exact spin-induced change of the periapsis advance: Kerr geodesic minus closed-form Schwarzschild."""
    return float(qe.einsteinpy_periapsis_advance(P, E, spin=SPIN)[0]) - qe.schwarzschild_advance_exact(P, E)


@slow
def test_einsteinpy_geodesic_agrees_with_closed_form(schwarzschild_geodesic):
    # two independent exact references; the step size (delta = 1) limits agreement to ~1e-4 rad
    assert schwarzschild_geodesic == pytest.approx(qe.schwarzschild_advance_exact(P, E), abs=5e-4)


@slow
def test_exact_prograde_spin_reduces_periapsis_advance(kerr_spin_effect):
    assert kerr_spin_effect < 0.0


@slow
def test_qgl_frame_dragging_has_the_exact_magnitude(kerr_spin_effect):
    effect = qe.qgl_spin_effect(P, E, SPIN)
    assert abs(effect) == pytest.approx(abs(kerr_spin_effect), rel=0.15)


@slow
def test_qgl_frame_dragging_has_the_exact_sign(kerr_spin_effect):
    assert np.sign(qe.qgl_spin_effect(P, E, SPIN)) == np.sign(kerr_spin_effect)
