"""Side-by-side comparison of the repo's cosmologies on one footing (qgl_compare)."""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import qgl_compare as qcmp
import qgl_kasner as qk


@pytest.fixture(scope="module")
def scenarios():
    return qcmp.all_scenarios()


@pytest.fixture(scope="module")
def result(scenarios):
    return qcmp.compare(scenarios)


def test_seven_scenarios_with_unique_names(scenarios):
    assert [s.name for s in scenarios] == ["kasner", "mixmaster", "pancake", "bounce", "lcdm", "toy", "ccbh"]


def test_normalisation_puts_the_fiducial_event_at_one_and_zero(scenarios):
    for sc in scenarios:
        tau_hat, K_hat = sc.normalised()
        assert K_hat[sc.i_fid] == pytest.approx(1.0, rel=1e-14), sc.name
        assert tau_hat[sc.i_fid] == pytest.approx(0.0, abs=1e-14), sc.name
        assert np.all(np.diff(tau_hat) > 0.0), sc.name
        assert np.all(np.isfinite(sc.K)) and np.all(sc.K >= 0.0), sc.name
        assert sc.T_K == pytest.approx(sc.K_ref ** -0.25)


def test_sensor_is_at_half_complexity_at_the_fiducial_event_and_physical_everywhere(scenarios, result):
    table, runs = result
    for sc in scenarios:
        df = runs[sc.name]
        assert np.interp(0.0, df["tau_hat"], df["C_Q"]) == pytest.approx(0.5, abs=0.02), sc.name
        assert np.all(np.abs(df["sigma_z"]) <= 1.0 + 1e-9), sc.name
        assert np.all((df["purity"] > 0) & (df["purity"] <= 1.0 + 1e-9)), sc.name
        assert np.all(np.isfinite(df[["K_hat", "C_Q", "n", "omega_c"]].to_numpy())), sc.name
        assert np.all(np.diff(df["tau_hat"]) > 0.0), sc.name


def test_finite_histories_stay_near_the_fiducial_curvature_and_singular_ones_do_not(result):
    table, _ = result
    t = table.set_index("scenario")
    for name in ("pancake", "bounce"):
        assert not t.loc[name, "singular_history"] and t.loc[name, "K_hat_max"] < 2.0
    for name in ("kasner", "mixmaster", "lcdm"):
        assert t.loc[name, "singular_history"] and t.loc[name, "K_hat_max"] / t.loc[name, "K_hat_min"] > 1e3


def test_kasner_summary_matches_the_closed_form(scenarios, result):
    table, _ = result
    sc = next(s for s in scenarios if s.name == "kasner")
    row = table.set_index("scenario").loc["kasner"]
    p = qk.kasner_exponents(2.0)
    K_ref = float(abs(qk.kretschmann_closed_form(p, 1.0)))
    assert row["K_ref"] == pytest.approx(K_ref, rel=1e-12)
    assert row["K_hat_max"] == pytest.approx((1.0 / 0.05) ** 4, rel=1e-9)                # K ~ t^-4
    assert row["max_dlnK_dtau"] == pytest.approx(4.0 * sc.T_K / 0.05, rel=2e-2)           # d ln K / d tau_hat at t_lo
    assert row["window"] == pytest.approx((1.0 - 0.05) / sc.T_K, rel=1e-12)


def test_the_three_flrw_histories_are_nearly_indistinguishable_to_the_sensor(result):
    _, runs = result
    grid = np.linspace(-1.5, 0.0, 40)
    c = {k: np.interp(grid, runs[k]["tau_hat"], runs[k]["C_Q"]) for k in ("lcdm", "toy", "ccbh")}
    assert np.max(np.abs(c["toy"] - c["lcdm"])) < 0.02 and np.max(np.abs(c["ccbh"] - c["lcdm"])) < 0.02
    # ... although the curvature histories themselves differ by tens of percent: C_Q ~ K^(p/6) is a weak function of K
    k = {n: np.interp(grid, runs[n]["tau_hat"], runs[n]["K_hat"]) for n in c}
    for name in ("toy", "ccbh"):
        gap = np.max(np.abs(k[name] / k["lcdm"] - 1.0))
        assert 0.01 < gap < 0.6, name


def test_finite_cases_are_gentler_than_the_singular_ones_in_curvature_time(result):
    t = result[0].set_index("scenario")
    assert t.loc["bounce", "max_dlnK_dtau"] < 5.0
    assert t.loc["pancake", "max_dlnK_dtau"] < t.loc["kasner", "max_dlnK_dtau"]
    assert t.loc["bounce", "omega_c_max"] < 0.2 * t.loc["kasner", "omega_c_max"]    # sensor frequency is never driven up


def test_plot_writes_a_png(tmp_path, scenarios, result):
    out = tmp_path / "cmp.png"
    qcmp.plot_comparison(str(out), scenarios, runs=result[1])
    assert out.exists() and out.stat().st_size > 10_000
