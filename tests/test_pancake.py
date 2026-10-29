"""The non-singular pancake Big Bang (qgl_pancake; Dechant, Lasenby & Hobson, arXiv:0809.4335).

The equations were derived independently of the paper and are checked here against its eqs. (25), (26), (27),
its scaling symmetry (29) and its finite-curvature claim.  The Kretschmann closed form is checked against a
brute-force Riemann tensor (the fixture from test_bkl.py, which skips without sympy).
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import qgl_bkl as bk
import qgl_pancake as qp
from test_bkl import brute  # noqa: F401  (fixture)

A0, B0, F0, M, LAM = 1.0, 3.0, 2.0, 0.5, 0.1


@pytest.fixture(scope="module")
def sol():
    return qp.pancake_solution(A0, B0, F0, M, LAM, t_max=5.0, n=300)


def states(s):
    return np.hstack([s["R"], s["Rd"], s["phi"][:, None], s["phid"][:, None]])


def test_slice_ricci_matches_bianchi9_and_is_stable_near_the_pancake():
    rng = np.random.default_rng(3)
    R = rng.uniform(0.3, 2.0, size=(20, 3))
    assert qp.slice_ricci(R) == pytest.approx(bk.ricci3(R / 2.0), rel=1e-12)
    r3 = qp.slice_ricci(np.array([1e-6, 3.0, 3.0]))
    assert r3[1] == pytest.approx(r3[2], rel=1e-12)                       # biaxial symmetry survives roundoff
    assert r3[1] == pytest.approx(4.0 / 9.0, rel=1e-9)                    # 4 / b0^2
    assert r3[0] == pytest.approx(0.0, abs=1e-12)


def test_series_satisfies_the_constraint_to_higher_order():
    c = qp.pancake_series(A0, B0, F0, M, LAM)
    res = [abs(qp.constraint(qp.series_state(c, t), M, LAM)) for t in (1e-2, 2e-2)]
    assert res[0] < 1e-5 and res[1] / res[0] > 3.0                        # vanishes as t -> 0 (O(t^2))
    for t in (1e-3, 1e-2):                                                # and the series solves the ODEs
        y = qp.series_state(c, t)
        assert qp.pancake_rhs(y, M, LAM)[3:6] == pytest.approx(
            [6.0 * c["a2"] * t, 2.0 * c["b2"], 2.0 * c["b2"]], rel=2e-3, abs=2e-5)


def test_constraint_is_preserved_along_the_solution(sol):
    assert max(abs(qp.constraint(s, M, LAM)) for s in states(sol)) < 1e-7


def test_parity_and_limits_at_the_pancake(sol):
    R = sol["R"]
    assert R[:, 0] == pytest.approx(-R[::-1, 0], abs=1e-14)               # R1 odd
    assert R[:, 1] == pytest.approx(R[::-1, 1], abs=1e-14)                # R2 even
    assert sol["phi"] == pytest.approx(sol["phi"][::-1], abs=1e-14)       # phi even
    assert R[:, 1] == pytest.approx(R[:, 2], rel=1e-12)                   # biaxial
    near = np.abs(sol["t"]) < 2e-3
    assert R[near, 0] == pytest.approx(A0 * sol["t"][near], rel=2e-5)     # R1 ~ a0 t
    assert R[near, 1] == pytest.approx(B0, rel=1e-5)
    assert sol["phi"][near] == pytest.approx(F0, rel=1e-5)
    # volume goes to zero while R2 stays finite
    vol = np.abs(R[:, 0] * R[:, 1] * R[:, 2])
    assert vol[np.argmin(np.abs(sol["t"]))] < 1e-3 * vol[-1]


def test_paper_field_equations_hold(sol):
    R, Rd, Rdd, ph, pd_ = sol["R"], sol["Rd"], sol["Rdd"], sol["phi"], sol["phid"]
    H = Rd / R
    V = 0.5 * M * M * ph ** 2
    rho, p = 0.5 * pd_ ** 2 + V, 0.5 * pd_ ** 2 - V
    r1, r2 = R[:, 0], R[:, 1]
    Hd2 = Rdd[:, 1] / r2 - H[:, 1] ** 2
    e25 = 2 * Hd2 + 3 * H[:, 1] ** 2 + p - LAM - (3 * r1 ** 2 / r2 ** 2 - 4) / r2 ** 2
    e26 = (2 * Rdd[:, 0] / r1 - H[:, 1] ** 2 + 2 * H[:, 0] * H[:, 1] + p - LAM
           + (5 * r1 ** 2 / r2 ** 2 - 4) / r2 ** 2)                       # 2 H1' + 2 H1^2 = 2 R1''/R1
    e27 = H[:, 1] ** 2 + 2 * H[:, 0] * H[:, 1] - rho - LAM - (r1 ** 2 / r2 ** 2 - 4) / r2 ** 2
    e28 = sol["Rdd"][:, 0] * 0 + (np.gradient(pd_, sol["t"]) + (H[:, 0] + 2 * H[:, 1]) * pd_ + M * M * ph)
    for e in (e25, e26, e27):
        assert np.max(np.abs(e)) < 1e-7
    mid = (np.abs(sol["t"]) > 1e-2)
    assert np.max(np.abs(e28[mid])) < 5e-3                                # finite-difference check of eq. 28


def test_scaling_symmetry_eq_29():
    s = 2.0
    base = qp.pancake_solution(A0, B0, F0, M, LAM, t_max=4.0, n=200)
    scaled = qp.pancake_solution(A0, B0 / s, F0, s * M, s * s * LAM, t_max=4.0 / s, n=200, t_start=1e-4 / s)
    # R_bar(t) = R(s t) / s  (so b0 -> b0 / s, a0 and f0 unchanged): compare at matching points t_bar = t / s
    checked = 0
    for t in (0.37, 1.1, 3.5):
        i_b = np.argmin(np.abs(base["t"] - t))
        tb = base["t"][i_b] / s
        j = np.argmin(np.abs(scaled["t"] - tb))
        if abs(scaled["t"][j] - tb) > 2e-3 * abs(tb):
            continue
        assert scaled["R"][j] == pytest.approx(base["R"][i_b] / s, rel=2e-3)
        assert scaled["phi"][j] == pytest.approx(base["phi"][i_b], rel=2e-3)
        checked += 1
    assert checked >= 2


def test_curvature_is_finite_and_matches_the_closed_form(sol):
    K = qp.pancake_kretschmann(sol)
    K0 = qp.kretschmann_at_pancake(B0, F0, M, LAM, a0=A0)
    near = np.abs(sol["t"]) < 1e-2
    assert K[near] == pytest.approx(K0, rel=2e-4)
    assert np.all(np.isfinite(K)) and np.max(np.abs(K)) < 10 * K0
    # not the paper's three coordinate components alone: 16 e^2 + (128 - 48 a0^2)/b0^4, with e = 2 b2 / b0
    c = qp.pancake_series(A0, B0, F0, M, LAM)
    assert K0 == pytest.approx(16.0 * (2.0 * c["b2"] / B0) ** 2 + (128.0 - 48.0 * A0 ** 2) / B0 ** 4, rel=1e-12)


@pytest.mark.parametrize("a0,b0,f0,m,lam", [(1.2, 3.0, 2.0, 0.5, 0.1), (0.6, 2.0, 1.5, 0.8, 0.0),
                                            (1.2, 18000.0, 13.0, 1.0 / 64000.0, 0.0)])
def test_closed_form_curvature_depends_on_the_opening_speed(a0, b0, f0, m, lam):
    # a0 != 1 and the paper's own parameters; an a0-free formula is off by 20% for the latter
    sol = qp.pancake_solution(a0, b0, f0, m, lam, t_max=0.02 * b0, t_min=-0.02 * b0, t_start=1e-3, n=120)
    K = qp.pancake_kretschmann(sol)
    K0 = qp.kretschmann_at_pancake(b0, f0, m, lam, a0=a0)
    near = np.abs(sol["t"]) < 1e-3 * b0                                   # K(t) - K(0) = O((t / b0)^2)
    assert K[near] == pytest.approx(K0, rel=2e-3)


def test_kretschmann_closed_form_against_brute_force_riemann(brute):
    c = qp.pancake_series(A0, B0, F0, M, LAM)
    t = 1e-3
    y = qp.series_state(c, t)
    d = qp.pancake_rhs(y, M, LAM)
    a, ad, add = y[:3] / 2.0, y[3:6] / 2.0, d[3:6] / 2.0
    assert bk.bianchi9_kretschmann(a, ad, add) == pytest.approx(brute(a, ad, add)[0], rel=1e-8)
    assert qp.kretschmann_at_pancake(B0, F0, M, LAM, a0=A0) == pytest.approx(brute(a, ad, add)[0], rel=1e-5)


def test_shear_diverges_like_one_over_t_squared(sol):
    s2 = qp.shear_squared(sol)
    near = (np.abs(sol["t"]) > 2e-4) & (np.abs(sol["t"]) < 2e-3)
    assert sol["t"][near] ** 2 * s2[near] == pytest.approx(1.0 / 3.0, rel=2e-3)
    assert s2[np.argmin(np.abs(sol["t"]))] > 1e6


def test_solution_input_validation():
    with pytest.raises(ValueError):
        qp.pancake_solution(A0, B0, F0, M, t_max=1e-6)


def test_sensor_runs_through_the_pancake(sol):
    df = qp.pancake_sensor_run(sol, -3.0, 3.0, rate_scale=5.0, n=150)
    K0 = qp.kretschmann_at_pancake(B0, F0, M, LAM, a0=A0)
    assert isinstance(df, pd.DataFrame) and {"t", "tau", "K", "C_Q", "n", "sigma_z", "purity"} <= set(df.columns)
    assert df["t"].min() == pytest.approx(-3.0, abs=0.2) and df["t"].max() == pytest.approx(3.0, abs=0.2)
    assert np.any(df["t"] == 0.0) and df.loc[df["t"] == 0.0, "K"].iloc[0] == pytest.approx(K0, rel=1e-12)
    assert np.all(np.diff(df["tau"]) >= 0.0)
    assert np.all(np.isfinite(df[["K", "C_Q", "n", "sigma_z", "omega_c"]].to_numpy()))
    assert np.all(np.abs(df["sigma_z"]) <= 1.0 + 1e-9) and np.all((df["purity"] > 0) & (df["purity"] <= 1.0 + 1e-9))
    # no divergence: the sensor sees a smooth, bounded drive through the pancake
    assert df["omega_c"].max() < 3.0 * df["omega_c"].min() + 50.0


def test_sensor_run_backwards_in_time_is_allowed(sol):
    df = qp.pancake_sensor_run(sol, 2.0, -2.0, rate_scale=3.0, n=100)
    assert df["t"].iloc[0] > df["t"].iloc[-1]
    assert np.all(np.diff(df["tau"]) >= 0.0)
