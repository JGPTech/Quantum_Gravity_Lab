"""Kasner vacuum cosmology, verified with EinsteinPy's symbolic tensors.

Skipped entirely when EinsteinPy is not installed.  All tests are fast (<~10 s).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("einsteinpy")
sp = pytest.importorskip("sympy")

import qgl_kasner as qk

R = sp.Rational
EXACT_U = [R(1), R(3, 2), R(2), R(5)]


# ---------------------------------------------------------------- exponents

@pytest.mark.parametrize("u", [1.0, 1.5, 2.0, 5.0, 10.0, 100.0])
def test_parametrisation_satisfies_both_kasner_conditions(u):
    p = qk.kasner_exponents(u)
    s1, s2 = qk.kasner_conditions(p)
    assert s1 == pytest.approx(1.0, abs=1e-12)
    assert s2 == pytest.approx(1.0, abs=1e-12)
    assert qk.is_vacuum_kasner(p)
    assert p[0] <= 0.0 <= p[1] <= p[2]


def test_special_values_of_the_parametrisation():
    assert qk.kasner_exponents(R(1)) == (R(-1, 3), R(2, 3), R(2, 3))
    assert qk.kasner_exponents(R(2)) == (R(-2, 7), R(3, 7), R(6, 7))
    p = qk.kasner_exponents(1.0e6)                       # u -> infinity approaches flat (0, 0, 1)
    assert p == pytest.approx((0.0, 0.0, 1.0), abs=1e-5)


def test_u_below_one_is_rejected():
    with pytest.raises(ValueError):
        qk.kasner_exponents(0.5)


def test_isotropic_exponents_are_not_vacuum():
    p = (R(1, 3), R(1, 3), R(1, 3))
    assert sum(p) == 1 and not qk.is_vacuum_kasner(p)    # sphere condition fails


# ---------------------------------------------------------------- symbolic curvature

@pytest.mark.parametrize("u", EXACT_U)
def test_ricci_vanishes_for_every_admissible_exponent_set(u):
    assert qk.kasner_curvature(qk.kasner_exponents(u))["ricci_is_zero"]


@pytest.mark.parametrize("u", EXACT_U)
def test_kretschmann_is_minus_16_p1p2p3_over_t4(u):
    p = qk.kasner_exponents(u)
    c = qk.kasner_curvature(p)
    assert sp.simplify(c["kretschmann"] - qk.kretschmann_closed_form(p, c["t"])) == 0
    assert sp.simplify(c["kretschmann"] * c["t"] ** 4).free_symbols == set()   # exactly t^-4
    assert c["kretschmann"].subs(c["t"], 1) > 0                                # p1 < 0 makes K positive


def test_flat_case_has_no_curvature():
    c = qk.kasner_curvature((0, 0, 1))
    assert c["ricci_is_zero"] and c["riemann_is_zero"]
    assert c["kretschmann"] == 0


@pytest.mark.parametrize("u", EXACT_U)
def test_any_other_vacuum_solution_is_genuinely_curved(u):
    c = qk.kasner_curvature(qk.kasner_exponents(u))
    assert not c["riemann_is_zero"]                      # vacuum but curved: pure Weyl curvature


def test_isotropic_non_vacuum_has_nonzero_ricci():
    c = qk.kasner_curvature((R(1, 3), R(1, 3), R(1, 3)))
    assert not c["ricci_is_zero"]


# ---------------------------------------------------------------- volume

@pytest.mark.parametrize("u", EXACT_U)
def test_volume_grows_linearly_in_t(u):
    assert qk.kasner_volume_exponent(qk.kasner_exponents(u)) == 1
