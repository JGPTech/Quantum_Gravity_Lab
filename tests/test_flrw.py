"""Toy flat-FLRW histories (qgl_flrw): LCDM limits, kinematics, horizon thermodynamics, Kretschmann, sensor."""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import qgl_flrw as fl

A = np.geomspace(0.02, 8.0, 60)
GRID = np.geomspace(0.05, 10.0, 400)


# ---------------------------------------------------------------- LCDM limits

def test_toy_recovers_lcdm_when_the_sigmoid_moves_to_the_far_past():
    assert fl.toy_E2(A, fl.ToyParams(0.3, 0.0, 0.0, 4.0)) == pytest.approx(fl.lcdm_E2(A, 0.3), rel=1e-14)
    assert fl.toy_E2(A, fl.ToyParams(0.3, 0.0, 1e-6, 4.0)) == pytest.approx(fl.lcdm_E2(A, 0.3), rel=1e-12)
    assert fl.toy_E2(A, fl.ToyParams(0.3, 1e-4, 1e-6, 4.0)) == pytest.approx(fl.lcdm_E2(A, 0.3, 1e-4), rel=1e-12)


@pytest.mark.parametrize("a_t,k", [(0.0, 4.0), (0.4, 4.0), (1.5, 9.0)])
def test_every_model_is_normalised_at_a_equal_one(a_t, k):
    assert fl.toy_E2(1.0, fl.ToyParams(0.3, 0.0, a_t, k)) == pytest.approx(1.0, rel=1e-14)
    assert fl.tsallis_E2(1.0, 0.9) == pytest.approx(1.0) and fl.pade_E2(1.0, -1.1) == pytest.approx(1.0)


def test_literature_models_reduce_to_lcdm():
    assert fl.tsallis_E2(A, 1.0) == pytest.approx(fl.lcdm_E2(A), rel=1e-13)
    assert fl.pade_E2(A, -1.0) == pytest.approx(fl.lcdm_E2(A), rel=1e-13)


def test_analytic_log_derivative_matches_numeric():
    p = fl.ToyParams(0.3, 1e-4, 0.5, 5.0)
    assert fl.toy_dlnE2_dlna(GRID, p) == pytest.approx(fl.dlnE2_dlna_numeric(lambda x: fl.toy_E2(x, p), GRID), rel=1e-6, abs=1e-8)


# ---------------------------------------------------------------- kinematics and distances

def test_cosmic_time_matches_the_lcdm_closed_form():
    Om = 0.3
    OL = 1.0 - Om
    t = fl.cosmic_time(lambda x: fl.lcdm_E2(x, Om), A[:20])
    exact = (2.0 / 3.0) / np.sqrt(OL) * np.arcsinh(np.sqrt(OL / Om) * A[:20] ** 1.5)
    assert t == pytest.approx(exact, rel=1e-9)


def test_luminosity_distance_matches_einstein_de_sitter():
    z = np.array([0.1, 0.5, 1.0, 2.0])
    d = fl.luminosity_distance(lambda x: fl.lcdm_E2(x, 1.0), z)
    assert d == pytest.approx(2.0 * (1.0 + z) * (1.0 - 1.0 / np.sqrt(1.0 + z)), rel=1e-10)


def test_deceleration_is_one_half_in_matter_domination_and_tends_to_minus_one():
    E2 = lambda x: fl.lcdm_E2(x, 0.3)
    q = fl.deceleration(E2, np.array([1e-3, 1e-2, 1e3]))
    assert q[0] == pytest.approx(0.5, abs=1e-3) and q[2] == pytest.approx(-1.0, abs=1e-6)
    assert fl.deceleration(E2, 1.0) == pytest.approx(-1.0 + 1.5 * 0.3, rel=1e-6)    # q0 = 3 Om/2 - 1


def test_distance_modulus_departs_from_lcdm_as_the_toy_departs():
    z = np.linspace(0.02, 1.5, 30)
    mu0 = fl.distance_modulus(lambda x: fl.lcdm_E2(x, 0.3), z)

    def rms(a_t):
        p = fl.ToyParams(0.3, 0.0, a_t, 4.0)
        return np.sqrt(np.mean((fl.distance_modulus(lambda x: fl.toy_E2(x, p), z) - mu0) ** 2))
    assert rms(1e-6) < 1e-9
    assert 1e-3 < rms(0.2) < rms(0.4) < rms(0.8)


# ---------------------------------------------------------------- horizon thermodynamics

def test_lcdm_obeys_the_second_law_and_horizon_entropy_grows():
    E2 = lambda x: fl.lcdm_E2(x, 0.3)
    assert fl.second_law_holds(E2, GRID)
    S = fl.horizon_entropy(E2, GRID)
    assert np.all(np.diff(S) > 0.0)
    assert S[-1] == pytest.approx(np.pi / 0.7, rel=1e-3)                  # de Sitter horizon pi / (G H_inf^2)


def test_entropy_rate_sign_is_positive_exactly_when_q_exceeds_minus_one():
    p = fl.ToyParams(0.3, 0.0, 1.0, 4.0)
    E2 = lambda x: fl.toy_E2(x, p)
    d = fl.toy_dlnE2_dlna(GRID, p)
    q = fl.deceleration(E2, GRID, d)
    sign = fl.entropy_rate_sign(E2, GRID, d)
    assert np.all((sign > 0) == (q > -1.0))
    assert q.min() < -1.0                                                  # this toy goes phantom ...
    assert not fl.second_law_holds(E2, GRID, d)                            # ... and breaks the second law


def test_mild_toy_keeps_the_second_law_and_a_strong_one_does_not():
    mild, strong = fl.ToyParams(0.3, 0.0, 0.4, 4.0), fl.ToyParams(0.3, 0.0, 0.7, 20.0)
    for p, ok in ((mild, True), (strong, False)):
        assert fl.second_law_holds(lambda x: fl.toy_E2(x, p), GRID, fl.toy_dlnE2_dlna(GRID, p)) is ok


def test_pade_model_with_q0_below_minus_one_violates_the_second_law():
    # the thermodynamic motivation of arXiv:2202.03300 requires q >= -1; its own best fit q_de0 ~ -1.02 is close to the edge
    assert fl.second_law_holds(lambda x: fl.pade_E2(x, -1.0), GRID)
    assert not fl.second_law_holds(lambda x: fl.pade_E2(x, -1.05), GRID)


# ---------------------------------------------------------------- Kretschmann

def test_kretschmann_de_sitter_and_matter_domination():
    H = 0.7
    assert fl.flrw_kretschmann(H, 0.0) == pytest.approx(24.0 * H ** 4, rel=1e-14)
    t = np.array([0.5, 2.0])                                               # a ~ t^(2/3): H = 2/(3t), H' = -2/(3t^2)
    assert fl.flrw_kretschmann(2.0 / (3.0 * t), -2.0 / (3.0 * t ** 2)) == pytest.approx(80.0 / (27.0 * t ** 4), rel=1e-13)


def test_kretschmann_formula_against_einsteinpy_symbolic():
    pytest.importorskip("einsteinpy")
    sp = pytest.importorskip("sympy")
    import qgl_curvature as qc
    t, x, y, z = sp.symbols("t x y z", positive=True)
    a = sp.Function("a")(t)
    K = qc.kretschmann_of_diagonal_metric([-1, a ** 2, a ** 2, a ** 2], (t, x, y, z))
    H = sp.diff(a, t) / a
    mine = 12 * ((sp.diff(H, t) + H ** 2) ** 2 + H ** 4)
    assert sp.simplify(K - mine) == 0


# ---------------------------------------------------------------- history and sensor

def test_history_table():
    E2 = lambda x: fl.lcdm_E2(x, 0.3)
    h = fl.flrw_history(E2, np.geomspace(0.05, 3.0, 40))
    assert {"a", "z", "t", "H", "q", "dS_sign", "K"} <= set(h.columns)
    assert np.all(np.diff(h["t"]) > 0) and np.all(np.diff(h["K"]) < 0)    # curvature falls as the universe expands
    assert np.all(h["dS_sign"] > 0)


def test_sensor_rides_the_lcdm_history():
    E2 = lambda x: fl.lcdm_E2(x, 0.3)
    df = fl.flrw_sensor_run(E2, 0.1, 4.0, n=80)
    assert np.all(np.diff(df["tau"]) > 0) and np.all(np.diff(df["C_Q"]) < 0)   # K falls, so C_Q falls
    assert df.loc[(df["a"] - 1.0).abs().idxmin(), "C_Q"] == pytest.approx(0.5, abs=0.05)   # K_trans = K today
    assert np.all(np.abs(df["sigma_z"]) <= 1.0 + 1e-9)
    assert np.all((df["purity"] > 0) & (df["purity"] <= 1.0 + 1e-9))
    assert np.all(np.isfinite(df[["K", "C_Q", "n", "omega_c"]].to_numpy()))


# ---------------------------------------------------------------- bounce toy

BOM, BRC = 0.3, 10.0


def test_bounce_exact_solution_satisfies_the_friedmann_equation():
    t = np.linspace(-3.0, 3.0, 41)
    a = fl.bounce_scale_factor(t, BOM, BRC)
    assert fl.bounce_hubble(t, BOM, BRC) ** 2 == pytest.approx(fl.bounce_E2(a, BOM, BRC), rel=1e-12, abs=1e-14)
    a_b = (BOM / BRC) ** (1.0 / 3.0)
    assert fl.bounce_scale_factor(0.0, BOM, BRC) == pytest.approx(a_b, rel=1e-14)
    assert fl.bounce_E2(a_b, BOM, BRC) == pytest.approx(0.0, abs=1e-14)
    assert fl.bounce_hubble(-t, BOM, BRC) == pytest.approx(-fl.bounce_hubble(t, BOM, BRC))   # contracting -> expanding


def test_bounce_hubble_dot_matches_a_finite_difference_and_the_bounce_value():
    t = np.array([-1.5, -0.2, 0.0, 0.3, 2.0])
    h = 1e-6
    num = (fl.bounce_hubble(t + h, BOM, BRC) - fl.bounce_hubble(t - h, BOM, BRC)) / (2 * h)
    assert fl.bounce_hubble_dot(t, BOM, BRC) == pytest.approx(num, rel=1e-7)
    assert fl.bounce_hubble_dot(0.0, BOM, BRC) == pytest.approx(1.5 * BRC, rel=1e-14)


def test_bounce_late_times_are_matter_dominated():
    t = np.array([50.0, 200.0])
    H, Hd = fl.bounce_hubble(t, BOM, BRC), fl.bounce_hubble_dot(t, BOM, BRC)
    assert -(1.0 + Hd / H ** 2) == pytest.approx(0.5, abs=1e-3)           # q -> 1/2


def test_bounce_curvature_is_finite_and_peaks_at_the_bounce():
    t = np.linspace(-2.0, 2.0, 401)
    K = fl.flrw_kretschmann(fl.bounce_hubble(t, BOM, BRC), fl.bounce_hubble_dot(t, BOM, BRC))
    assert K[200] == pytest.approx(27.0 * BRC ** 2, rel=1e-12)
    assert np.argmax(K) == 200 and np.all(np.isfinite(K))


def test_sensor_crosses_the_bounce():
    df = fl.bounce_sensor_run(2.0, BOM, BRC, n=81)
    assert np.any(df["t"] == 0.0) and df["K"].max() == pytest.approx(27.0 * BRC ** 2, rel=1e-12)
    assert np.all(np.diff(df["tau"]) > 0) and np.all(np.isfinite(df[["K", "C_Q", "n", "omega_c"]].to_numpy()))
    assert df.loc[df["t"] == 0.0, "C_Q"].iloc[0] == pytest.approx(0.5, abs=1e-12)    # K = K_trans at the bounce
    assert np.all(np.abs(df["sigma_z"]) <= 1.0 + 1e-9)
