"""Tests for qgl_tiling.py (Chair44 label cubes) and qgl_tiling_sensor.py (JC sensor along lines through a tiling).

They need the Spectre repository (SPECTRE_DIR or ../spectre) and are skipped without it."""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import qgl_tiling as T

pytestmark = pytest.mark.skipif(not T.available(), reason="Spectre repository not found (set SPECTRE_DIR)")


def test_supertile_has_the_cells_and_notch_sum_of_the_paper():
    """A depth-n supertile has 7 * 8^n cells and its notch vectors sum to 7 * 2^n (1,1,1)  (2^n against 8^(n/2))."""
    S = T._load()
    for n in range(1, 5):
        cube = T.supertile_labels(n)
        ok = cube >= 0
        assert ok.sum() == 7 * 8 ** n
        s = T._STATE["notch"][cube[ok]].astype(int).sum(axis=0)
        assert tuple(s) == (7 * 2 ** n,) * 3


def test_label_cubes_agree_cell_for_cell_with_the_tile_level_code():
    sys.path.insert(0, T.SPECTRE_DIR)
    import chair44_slice as CS
    n = 3
    occ, tile = CS.occupancy(n)
    N = CS.notches(n)
    cube = T.supertile_labels(n)
    ok = cube >= 0
    assert np.array_equal(occ, ok)
    assert np.array_equal(N[tile[occ]], T._STATE["notch"][cube[ok]].astype(N.dtype))


def test_pyramid_observable_has_zero_mean_and_axis_line_sums_are_bounded():
    """The 192 heights sum to zero.  Along the coordinate axes the line sums of the pyramid observable stay bounded
    (here <= 3 * 36) while along the body diagonal they grow linearly, which is the direction dependence of the paper."""
    assert abs(T.pyramid_per_native_cell().sum()) < 1e-12
    cube = T.label_cube(5, 7)
    F = T.field(cube, "pyramid").astype(np.float64)
    assert abs(F.mean()) < 1e-3          # exactly 0 only in the limit; a finite cube is balanced to ~1e-5
    for ax in range(3):
        c = np.cumsum(F, axis=ax)
        win = np.take(c, np.arange(64, 128), axis=ax) - np.take(c, np.arange(0, 64), axis=ax)
        assert np.abs(win).max() <= 108.0
    diag = sum(F[t:64 + t, t:64 + t, t:64 + t] for t in range(64))
    assert np.abs(diag).max() >= 36.0 * 64 * 0.9          # ballistic: sum ~ 36 L


def test_sensor_phase_follows_the_dispersive_formula():
    """Engine phase == -(dt/Delta) sum (g_i^2 - g0^2) for small modulation (the dispersive picture the experiment uses)."""
    import qgl_tiling_sensor as TS
    cfg = TS.sensor_config()
    dt, eps = 2.0, 0.02
    m = np.random.default_rng(0).standard_normal(8)
    c = TS.run_sensor(m, cfg, eps, dt, (8,))
    ref = TS.run_sensor(np.zeros(8), cfg, 0.0, dt, (8,))
    D = cfg.omega_atom_0 - cfg.omega_cavity_0
    pred = -dt / D * np.sum((cfg.g_0 * (1 + eps * m)) ** 2 - cfg.g_0 ** 2)
    assert abs(np.angle(c[0] / ref[0]) - pred) < 0.01


def test_linear_sensor_keeps_coherence_along_an_axis_and_loses_it_for_a_shuffled_decoration():
    """The result of qgl_tiling_sensor.py in its cleanest form: frequency-modulated sensor, pyramid observable,
    along the z axis.  Tiling: coherence rises towards 1 (bounded Birkhoff sums); shuffled control: it decays."""
    import qgl_tiling_sensor as TS
    r = TS.experiment("pyramid", depth=7, lengths=(8, 32, 64), n_lines=250, modulate="omega_a",
                      directions={"z": (0.0, 0.0, 1.0)}, verbose=False)["directions"]["z"]
    assert r["tiling"]["R"][-1] > 0.85
    assert r["shuffled"]["R"][-1] < 0.35
    assert r["tiling"]["var_over_L"][-1] < 0.05 and abs(r["shuffled"]["var_over_L"][-1] - 1.0) < 0.2


def test_body_diagonal_is_worse_than_random():
    """The prediction 'every line dephases more slowly' is false: along (1,1,1) the pyramid sum is ballistic."""
    import qgl_tiling_sensor as TS
    r = TS.experiment("pyramid", depth=7, lengths=(8, 32), n_lines=250, modulate="omega_a",
                      directions={"d": (1.0, 1.0, 1.0)}, verbose=False)["directions"]["d"]
    assert r["tiling"]["var_over_L"][-1] > 5.0
    assert r["tiling"]["R"][0] < r["shuffled"]["R"][0] - 0.1     # dephases faster at short range; later 'revivals' are phase wrapping
