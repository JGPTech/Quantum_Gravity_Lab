"""Cosmologically-coupled black hole toy (qgl_ccbh)."""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import qgl_ccbh as cc
import qgl_flrw as fl

A = np.geomspace(2e-3, 1.0, 40)


def test_single_black_hole_growth_and_spin():
    a = np.array([0.25, 0.5, 1.0])
    m = cc.bh_mass(a, 0.25, 10.0, 3.0)
    assert m == pytest.approx(10.0 * (a / 0.25) ** 3)
    h = 1e-6                                                              # dm/dt = k H m  <=>  dm/dln a = k m
    dm_dlna = (cc.bh_mass(a * np.exp(h), 0.25, 10.0, 3.0) - cc.bh_mass(a * np.exp(-h), 0.25, 10.0, 3.0)) / (2 * h)
    assert dm_dlna == pytest.approx(3.0 * m, rel=1e-8)
    assert cc.bh_spin(2.0, 1.0, 1.0, 3.0) == pytest.approx(0.015625)      # Ghodla et al.: ~0.016
    assert cc.density_exponent(3.0) == 0.0 and cc.density_exponent(0.0) == -3.0


def test_sfrd_madau_dickinson_shape():
    z = np.linspace(0.0, 8.0, 801)
    psi = cc.sfrd_madau_dickinson(1.0 / (1.0 + z))
    assert psi[0] == pytest.approx(0.015 / (1.0 + (1.0 / 2.9) ** 5.6), rel=1e-12)
    assert 1.5 < z[np.argmax(psi)] < 2.5                                  # cosmic noon
    assert psi[-1] < psi.max() / 5.0


def test_zero_efficiency_is_exactly_lcdm():
    bg = cc.ccbh_background(0.3, 0.0, 3.0)
    assert bg.OL == pytest.approx(0.7, abs=1e-12)
    assert bg.E2(A) == pytest.approx(fl.lcdm_E2(A, 0.3), rel=1e-9)


def test_uncoupled_conversion_is_also_lcdm():
    # k = 0: converted mass does not grow, so it stays matter-like and the expansion history is unchanged
    bg = cc.ccbh_background(0.3, 2.0, 0.0)
    assert bg.OL == pytest.approx(0.7, abs=1e-10)
    assert bg.E2(A) == pytest.approx(fl.lcdm_E2(A, 0.3), rel=1e-8)


@pytest.mark.parametrize("eps,k", [(0.5, 3.0), (1.0, 3.0), (1.0, 2.0)])
def test_normalisation_and_ordering(eps, k):
    bg = cc.ccbh_background(0.3, eps, k)
    assert bg.E2(1.0) == pytest.approx(1.0, abs=1e-9)
    assert bg.converted > 0.0 and bg.omega_bh > 0.0
    assert bg.OL < 0.7                                                    # black holes take over part of the dark term
    assert np.all(bg.E2(A) > 0.0)


def test_k3_physical_density_is_the_undiluted_formation_integral():
    eps = 1.0
    bg = cc.ccbh_background(0.3, eps, 3.0)
    x = np.log(bg.a)
    H = np.sqrt(bg.E2(bg.a))
    # k = 3: B / a^3 = eps * int psi / a'^3 dt = eps * int psi / (a'^3 H) dln a'  (each epoch at its physical density)
    integrand = cc.sfrd_code_units(bg.a, bg.h) / (H * bg.a ** 3)
    expected = eps * np.trapz(integrand, x) if hasattr(np, "trapz") else eps * np.trapezoid(integrand, x)
    assert bg.omega_bh == pytest.approx(expected, rel=2e-3)
    assert bg.omega_bh > bg.converted                                     # undoing dilution adds density


def test_k3_dark_term_is_flat_after_formation_ends():
    # late times: psi is tiny, so B/a^3 changes very little between a = 0.7 and 1 compared with its value
    bg = cc.ccbh_background(0.3, 1.0, 3.0)
    d = bg._interp(np.array([0.7, 1.0]), bg.B) / np.array([0.7, 1.0]) ** 3
    assert abs(d[1] / d[0] - 1.0) < 0.03


def test_required_efficiency_hits_the_target_and_exceeds_one():
    eps = cc.required_efficiency(0.3, 3.0)
    assert cc.ccbh_background(0.3, eps, 3.0).omega_bh == pytest.approx(0.7, rel=1e-6)
    assert eps > 1.0                                                      # more than all stellar mass: no free lunch


def test_distance_modulus_departure_grows_with_efficiency():
    z = np.linspace(0.02, 1.5, 25)
    mu0 = fl.distance_modulus(lambda x: fl.lcdm_E2(x, 0.3), z)
    rms = [np.sqrt(np.mean((fl.distance_modulus(cc.ccbh_background(0.3, e, 3.0).E2, z) - mu0) ** 2))
           for e in (0.0, 0.5, 1.0)]
    assert rms[0] < 1e-9 and 0.0 < rms[1] < rms[2]
