"""Sweep over the Mixmaster starting parameter u (qgl_bkl_sweep)."""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import qgl_bkl as bk
import qgl_bkl_sweep as sw
import qgl_kasner as qk

U = [1.3, 3.7, 8.5]


@pytest.fixture(scope="module")
def table():
    return sw.mixmaster_sweep(U)


def test_weyl_coefficient_closed_form():
    assert sw.kasner_weyl_coefficient(1.0) == pytest.approx(64.0 / 27.0, rel=1e-14)
    for u in (1.0, 2.5, 7.0):
        p = qk.kasner_exponents(u)
        assert sw.kasner_weyl_coefficient(u) == pytest.approx(float(qk.kretschmann_closed_form(p, 1.0)), rel=1e-13)
    c = [sw.kasner_weyl_coefficient(u) for u in (1.0, 2.0, 5.0, 50.0, 500.0)]
    assert np.all(np.diff(c) < 0.0) and c[-1] < 1e-2                      # -> flat (0, 0, 1) as u -> infinity


def test_clean_t0_sets_the_first_epoch_curvature_term():
    for u in (1.3, 3.7, 12.0):
        t0 = sw.clean_t0(u, 1.0, 1e-6)
        p1 = abs(qk.kasner_exponents(u)[0])
        assert (1.0 / t0) ** (4.0 * p1) / 2.0 == pytest.approx(1e-6, rel=1e-9)
    assert sw.clean_t0(12.0) > sw.clean_t0(3.7) > sw.clean_t0(1.3)       # weaker p1 needs a longer clean epoch


def test_fixed_t0_loses_the_first_epoch_at_large_u_and_clean_t0_restores_it():
    u = 8.5
    p = np.array(qk.kasner_exponents(u))
    with pytest.warns(RuntimeWarning):
        bad_mm = bk.Mixmaster(u, 1e6, 1.0, 1e-60)
    bad = bad_mm.epochs(n_grid=60000, threshold=2e-3)[0]
    assert np.max(np.abs(np.sort(bad["v"]) - np.sort(p))) > 0.02          # not the requested epoch
    good = bk.Mixmaster(u, sw.clean_t0(u), 1.0, 1e-60).epochs(n_grid=60000, threshold=2e-3)[0]
    assert np.max(np.abs(np.sort(good["v"]) - np.sort(p))) < 5e-3


def test_epochs_follow_the_landau_lifshitz_map_and_the_analytic_curvature(table):
    assert np.all(table["exponent_error"] < 5e-3)
    assert table["K_t4_measured"].to_numpy() == pytest.approx(table["K_coeff"].to_numpy(), rel=0.02)
    assert np.all(table["n_epochs"] == table["n_bounces"] + 1)
    assert np.all(table["n_bounces"] >= 1)


def test_weaker_epochs_bounce_later_and_more_gently(table):
    assert np.all(np.diff(table["K_coeff"]) < 0.0)
    assert np.all(np.diff(table["first_bounce_t"]) > 0.0)                 # slower p1 -> later first bounce
    assert np.all(np.diff(table["bounce_strength"]) < 0.0)


def test_sensor_response_orders_with_curvature_strength(table):
    assert np.all(np.diff(table["C_Q_at_bounce"]) < 0.0)                  # K_ref = 1/t_p^4: weaker K, smaller C_Q
    assert np.all(table["C_Q_start"] < table["C_Q_at_bounce"]) and np.all(table["C_Q_at_bounce"] < table["C_Q_end"])
    assert np.all(np.abs(table["sigma_z_final"]) <= 1.0 + 1e-9)
    assert np.all((table["purity_min"] > 0) & (table["purity_min"] <= 1.0 + 1e-9))
    assert np.all(np.isfinite(table[["tau_span", "omega_c_max", "n_final"]].to_numpy()))


def test_sensor_can_be_skipped():
    df = sw.mixmaster_sweep([2.5], sensor=False)
    assert "C_Q_at_bounce" not in df.columns and len(df) == 1


def test_mixmaster_warns_about_an_unclean_first_epoch_and_not_about_a_clean_one():
    with pytest.warns(RuntimeWarning, match="not clean"):
        bk.Mixmaster(8.5, 1e6, 1.0, 1e-60)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bk.Mixmaster(3.7, 1e6, 1.0, 1e-60)
        bk.Mixmaster(8.5, bk.clean_t0(8.5), 1.0, 1e-60)
    assert bk.first_epoch_term(8.5, bk.clean_t0(8.5), 1.0) == pytest.approx(1e-6, rel=1e-9)
