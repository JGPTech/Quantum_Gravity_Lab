"""Physics and regression tests for the QGL core and the precession/ensemble tools.

Run from the repository root:  python -m pytest -q
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from unified_sandbox_core import Config, canonical_periapsis_ic, unified_equations_of_motion
from qgl_precession import (configure_orbit, gr_1pn_shift, measure_precession,
                            solve_complexity_power)
import qgl_ensemble as ens


def cfg_for(p=100.0, e=0.6, **kw):
    c = configure_orbit(Config(), p, e)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_newtonian_limit_has_no_precession():
    c = cfg_for(alpha_gr=0.0)
    m = measure_precession(c, n_orbits=4)
    assert m["ok"]
    assert abs(m["shift"]) < 1e-8          # closed Kepler ellipse


def test_precession_scales_as_inverse_c_squared():
    # alpha_gr ~ 1/c^2 and the 1PN shift ~ 1/c^2: the ratio must not depend on c
    r = []
    for c_light in (10.0, 20.0):
        c = Config()
        c.c = c_light                       # set c first: alpha_GR is derived from it
        configure_orbit(c, 100.0, 0.6)
        r.append(measure_precession(c, n_orbits=3)["ratio"])
    assert r[0] == pytest.approx(r[1], rel=0.05)


def test_measurement_is_orbit_to_orbit_stable():
    m = measure_precession(cfg_for(), n_orbits=6)
    assert m["ok"] and m["n_peri"] >= 6
    assert m["scatter"] < 1e-8 * 1e2        # ~1e-10 in practice
    assert m["scatter"] < 1e-3 * abs(m["shift"])


def test_shift_has_gr_sign_and_order_of_magnitude():
    m = measure_precession(cfg_for(), n_orbits=4)
    assert m["shift"] > 0                   # prograde advance
    assert 0.3 < m["ratio"] < 1.7


def test_theory_value():
    c = cfg_for(100.0, 0.6)
    assert gr_1pn_shift(c) == pytest.approx(6 * np.pi / (c.c ** 2 * 100.0))


@pytest.mark.parametrize("p,e", [(60.0, 0.6), (150.0, 0.3)])
def test_solve_complexity_power_hits_gr(p, e):
    c = cfg_for(p, e)
    cp = solve_complexity_power(c, n_orbits=3)
    assert cp is not None
    assert measure_precession(c, n_orbits=3)["ratio"] == pytest.approx(1.0, abs=1e-5)


def test_solve_restores_config_when_unbracketed():
    c = cfg_for(complexity_power=0.7)
    assert solve_complexity_power(c, lo=0.05, hi=0.1, n_orbits=2) is None
    assert c.complexity_power == 0.7


def test_energy_conserved_without_gr_term():
    from scipy.integrate import solve_ivp
    c = cfg_for(alpha_gr=0.0)
    r0, v0 = canonical_periapsis_ic(c)
    sol = solve_ivp(lambda t, y: unified_equations_of_motion(t, y, c), [0, 2000.0],
                    list(r0) + list(v0), method="DOP853", rtol=1e-11, atol=1e-13)
    r = np.linalg.norm(sol.y[:3], axis=0)
    v2 = (sol.y[3:] ** 2).sum(axis=0)
    E = 0.5 * v2 - c.G * c.M / r
    assert np.ptp(E) < 1e-9 * abs(E[0])


def test_ensemble_grid_and_determinism():
    a = ens.main(["--p", "80", "120", "--e", "0.3", "--n-orbits", "3", "--workers", "1"])
    b = ens.main(["--p", "80", "120", "--e", "0.3", "--n-orbits", "3", "--workers", "1"])
    assert list(a["p"]) == [80.0, 120.0]
    assert np.allclose(a["ratio"], b["ratio"], rtol=0, atol=1e-12)
    # larger p -> smaller absolute advance
    assert a["shift"].iloc[0] > a["shift"].iloc[1] > 0


def test_ensemble_random_seed_reproducible():
    a = ens.main(["--random", "3", "--seed", "5", "--n-orbits", "3", "--workers", "1"])
    b = ens.main(["--random", "3", "--seed", "5", "--n-orbits", "3", "--workers", "1"])
    c = ens.main(["--random", "3", "--seed", "6", "--n-orbits", "3", "--workers", "1"])
    assert np.array_equal(a["p"], b["p"])
    assert not np.array_equal(a["p"], c["p"])


def test_ensemble_solve_mode_gives_unit_ratio():
    df = ens.main(["--p", "100", "--e", "0.5", "--n-orbits", "3", "--solve", "--workers", "1"])
    assert df["ratio"].iloc[0] == pytest.approx(1.0, abs=1e-5)
