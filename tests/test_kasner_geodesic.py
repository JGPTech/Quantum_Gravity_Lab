"""Timelike geodesics of the Kasner metric (qgl_kasner.kasner_geodesic).

Skipped when EinsteinPy is not installed (it supplies the independent
Christoffel-symbol check).  All tests are fast.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("einsteinpy")
sp = pytest.importorskip("sympy")

import qgl_kasner as qk

R = sp.Rational
P_EXACT = qk.kasner_exponents(R(2))                 # (-2/7, 3/7, 6/7)
P = tuple(float(x) for x in P_EXACT)
K_MOM = np.array([0.3, -0.2, 0.5])


# ---------------------------------------------------------------- simple cases

@pytest.mark.parametrize("t0,t_end", [(2.0, 5.0), (5.0, 2.0)])
def test_comoving_observer_stays_put_and_ages_with_t(t0, t_end):
    g = qk.kasner_geodesic(P, (0, 0, 0), t0, t_end, n_out=50)
    assert np.allclose(g["x"], 0.0, atol=1e-12)
    assert g["tau"] == pytest.approx(np.abs(g["t"] - t0), abs=1e-9)
    assert np.all(g["speed"] == 0.0)


@pytest.mark.parametrize("t0,t_end", [(2.0, 5.0), (5.0, 2.0)])
def test_moving_observer_ages_less_than_coordinate_time(t0, t_end):
    g = qk.kasner_geodesic(P, K_MOM, t0, t_end, n_out=200)
    assert np.all(np.diff(g["tau"]) > 0.0)               # tau increases on both branches
    assert 0.0 < g["tau"][-1] < abs(t_end - t0)


def test_worldline_is_unit_timelike():
    """-(dt/dtau)^2 + sum g_ii (dx^i/dtau)^2 = -1, from finite differences of the integrated path."""
    g = qk.kasner_geodesic(P, K_MOM, 3.0, 8.0, n_out=4000)
    d = lambda a: np.diff(a, axis=0)
    dtau = d(g["tau"])
    tm = 0.5 * (g["t"][1:] + g["t"][:-1])
    ds2 = -(d(g["t"]) / dtau) ** 2 + np.sum(tm[:, None] ** (2 * np.array(P)) * (d(g["x"]) / dtau[:, None]) ** 2, axis=1)
    assert ds2 == pytest.approx(-1.0, abs=1e-5)


def test_flat_exponents_give_straight_lines_in_minkowski():
    """(0,0,1) is Minkowski in Milne coordinates: T = t cosh z, Z = t sinh z.  Geodesics are straight."""
    g = qk.kasner_geodesic((0.0, 0.0, 1.0), (0.2, -0.1, 0.7), 1.0, 6.0, n_out=300)
    T = g["t"] * np.cosh(g["x"][:, 2])
    Z = g["t"] * np.sinh(g["x"][:, 2])
    for series in (T, Z, g["x"][:, 0], g["x"][:, 1]):
        slope, icpt = np.polyfit(g["tau"], series, 1)
        assert series == pytest.approx(slope * g["tau"] + icpt, abs=1e-7)


# ---------------------------------------------------------------- independent check

def _christoffels():
    """Gamma^mu_{alpha beta}(t) from EinsteinPy's symbolic Christoffel symbols, as a numeric function."""
    from einsteinpy.symbolic import ChristoffelSymbols
    metric, (t, *_ ) = qk.kasner_metric(P_EXACT)
    gam = ChristoffelSymbols.from_metric(metric).tensor()
    fns = [[[sp.lambdify(t, gam[m, a, b], "numpy") for b in range(4)] for a in range(4)] for m in range(4)]
    return lambda tt: np.array([[[float(fns[m][a][b](tt)) for b in range(4)] for a in range(4)] for m in range(4)])


@pytest.mark.parametrize("branch", [+1, -1])
def test_trajectory_solves_the_geodesic_equation_with_einsteinpy_christoffels(branch):
    """du^mu/dtau + Gamma^mu_{ab} u^a u^b = 0, with Gamma from EinsteinPy and u from the conserved momenta."""
    Gam = _christoffels()
    p, k = np.array(P), K_MOM

    def u(t):                                            # branch +1 expanding, -1 contracting (t decreasing)
        gamma = np.sqrt(1.0 + np.sum(k ** 2 * t ** (-2 * p)))
        return np.concatenate([[branch * gamma], k * t ** (-2 * p)])

    for t in (1.5, 3.0, 7.0):
        h = 1e-6 * t
        du_dt = (u(t + h) - u(t - h)) / (2 * h)
        u_now = u(t)
        du_dtau = du_dt * u_now[0]                       # dt/dtau = u^t
        accel = np.einsum("mab,a,b->m", Gam(t), u_now, u_now)
        scale = max(np.linalg.norm(accel), 1e-12)
        assert np.linalg.norm(du_dtau + accel) / scale < 1e-6


def test_speed_approaches_light_speed_at_the_singularity():
    g = qk.kasner_geodesic(P, K_MOM, 1.0, 1e-4, n_out=400)
    assert g["speed"][-1] > 0.999
    assert g["speed"][-1] > g["speed"][len(g["speed"]) // 2]


def test_nonpositive_times_are_rejected():
    with pytest.raises(ValueError):
        qk.kasner_geodesic(P, K_MOM, 0.0, 1.0)
