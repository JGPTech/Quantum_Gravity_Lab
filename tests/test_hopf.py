"""qgl_hopf: the pancake as a Berger sphere (Hopf fibre scaled by s = R_1/R_2), followed along the numerical solution."""
import math

import numpy as np
import pytest

import qgl_hopf as hp
import qgl_horizon as hz
import qgl_pancake as qp

SOL = dict(a0=1.0, b0=3.0, f0=2.0, m=0.5)


@pytest.fixture(scope="module")
def sol():
    return qp.pancake_solution(SOL["a0"], SOL["b0"], SOL["f0"], SOL["m"], t_max=6.0, n=600)


def test_solution_is_biaxial(sol):
    b = hp.berger_parameter(sol)
    assert np.all(b["t"] > 0)
    assert np.allclose(sol["R"][:, 1], sol["R"][:, 2], rtol=1e-12)


def test_squashing_is_a0_t_over_b0_near_the_pancake(sol):
    b = hp.berger_parameter(sol)
    early = b["t"] < 1e-2
    assert early.sum() > 5
    assert np.allclose(b["s"][early], SOL["a0"] * b["t"][early] / SOL["b0"], rtol=2e-2)


def test_squashing_passes_through_the_round_sphere(sol):
    b = hp.berger_parameter(sol)
    assert b["s"][0] < 1e-3 and b["s"].max() > 1.0       # crosses the round sphere (this solution peaks near 1.44, then falls back)


def test_slice_ricci_equals_berger_closed_form_along_solution(sol):
    b = hp.berger_parameter(sol)
    got = hp.slice_ricci_along(sol)
    want = hp.berger_ricci_closed(b["s"], b["R2"])
    assert np.allclose(got, want, rtol=1e-9, atol=1e-12 / SOL["b0"] ** 2)


def test_scalar_curvature_is_finite_at_the_pancake(sol):
    """Scalar 8/R_2^2 - 2 s^2/R_2^2 -> 8/b0^2: nothing diverges in the slice as the fibre collapses."""
    ric = hp.slice_ricci_along(sol)
    scalar = ric.sum(axis=1)
    assert scalar[0] * SOL["b0"] ** 2 == pytest.approx(8.0, rel=1e-3)
    assert np.all(np.isfinite(ric)) and abs(ric[0, 0]) * SOL["b0"] ** 2 < 1e-5


def test_paper_parameters_keep_the_slice_curvature_finite():
    p = qp.paper_parameters()
    sol = qp.pancake_solution(p["a0"], p["b0"], p["f0"], p["m"], t_max=1.0, n=60)
    scalar = hp.slice_ricci_along(sol).sum(axis=1)
    assert scalar[0] * p["b0"] ** 2 == pytest.approx(8.0, rel=1e-6)


def test_efolds_per_fibre_turn_tends_to_two_pi_a0(sol):
    b = hp.berger_parameter(sol)
    e = hp.efolds_per_turn(b["R1"], b["t"])
    assert e[0] == pytest.approx(2 * math.pi * SOL["a0"], rel=1e-2)
    assert e.max() > e[0]                     # R_1 / t is not constant once the fibre opens up (this solution: up to about 8.7)


def test_efolds_per_turn_matches_the_horizon_winding():
    """Between t_in and t_in e^N a ray winds N / (2 pi a0) turns at leading order."""
    a0 = 1.3
    n_e = 2 * math.pi * a0 * 2.0              # two turns
    assert hz.fibre_log_coefficient(a0) * n_e / hz.FIBRE_PERIOD == pytest.approx(2.0, rel=1e-12)
    assert hp.fibre_turns_leading(a0, math.exp(n_e), 1.0) == pytest.approx(2.0, rel=1e-12)


def test_fibre_winding_matches_leading_order_at_small_t(sol):
    h = hz.pancake_horizons(sol)
    b = hp.berger_parameter(sol)
    early = b["t"] < 1e-2
    lead = hp.fibre_turns_leading(SOL["a0"], b["t"][early], b["t"][0])
    assert np.allclose(h["fibre_turns"][early][1:], lead[1:], rtol=2e-2)


def test_berger_closed_form_unit_sphere():
    assert np.allclose(hp.berger_ricci_closed(1.0, 1.0), [2, 2, 2])
    assert hp.berger_ricci_closed(0.0, 2.0).sum() == pytest.approx(8 / 4)


def test_figure_is_written(tmp_path):
    path = tmp_path / "hopf.png"
    hp.plot_hopf(str(path))
    assert path.stat().st_size > 20000
