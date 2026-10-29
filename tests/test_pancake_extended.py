"""Pancake extensions: the paper's realistic parameters, triaxial perturbations, and winding geodesics."""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import qgl_bkl as bk
import qgl_pancake as qp

P = qp.paper_parameters()
KW = {k: P[k] for k in ("a0", "b0", "f0", "m", "Lam", "kappa")}


@pytest.fixture(scope="module")
def paper_sol():
    return qp.pancake_solution(**KW, t_max=3e6, t_min=-1e3, t_start=1e-3, n=400)


# ---------------------------------------------------------------- the paper's realistic parameters

def test_paper_parameters():
    assert (P["a0"], P["b0"], P["f0"], P["kappa"]) == (1.2, 18000.0, 13.0, 1.0)
    assert P["m"] == pytest.approx(1.0 / 64000.0) and P["Lam"] == 0.0


def test_paper_run_is_consistent_and_inflates_and_isotropises(paper_sol):
    s = paper_sol
    scale = []
    for i in range(0, len(s["t"]), 10):
        y = np.concatenate([s["R"][i], s["Rd"][i], [s["phi"][i], s["phid"][i]]])
        H = s["Rd"][i] / s["R"][i]
        terms = abs(H[0] * H[1]) + abs(H[1] * H[2]) + abs(H[0] * H[2]) + 0.5 * abs(qp.slice_ricci(s["R"][i])).sum()
        scale.append(abs(qp.constraint(y, P["m"], P["Lam"], P["kappa"])) / terms)
    assert max(scale) < 1e-8                                              # constraint, relative to its own terms
    assert qp.efolds(s) > 40.0                                            # scalar-driven inflation
    assert s["R"][-1, 0] / s["R"][-1, 1] == pytest.approx(1.0, abs=5e-3)  # the field isotropises R1 / R2
    i_roll = int(np.argmin(np.abs(s["t"] - 1e5)))
    assert 10.0 < s["phi"][i_roll] < 13.0                                 # slow roll from f0 = 13 ...
    assert abs(s["phi"][-1]) < 0.1 * P["f0"]                              # ... then it oscillates about 0 with falling amplitude
    assert s["R"][np.argmin(np.abs(s["t"])), 0] < 1e-2 and s["R"][-1, 1] > 1e20


def test_paper_curvature_at_the_pancake_is_finite_and_matches_the_closed_form(paper_sol):
    K = qp.pancake_kretschmann(paper_sol)
    K0 = qp.kretschmann_at_pancake(P["b0"], P["f0"], P["m"], P["Lam"], P["kappa"], P["a0"])
    near = np.abs(paper_sol["t"]) < 18.0
    assert K[near] == pytest.approx(K0, rel=2e-3)
    assert K0 == pytest.approx(8.35e-16, rel=2e-2) and np.all(np.isfinite(K))
    # leaving a0 out (the earlier closed form) would be 24% too large here
    assert qp.kretschmann_at_pancake(P["b0"], P["f0"], P["m"], a0=1.0) > 1.2 * K0


# ---------------------------------------------------------------- triaxial perturbation

def test_perturbation_sets_the_ratio_and_restores_the_constraint():
    y0 = qp.pancake_state_at(1e3, **KW)
    y = qp.perturb_triaxial(y0, 1e-4, P["m"], P["Lam"], P["kappa"])
    assert y[2] / y[1] - 1.0 == pytest.approx(1e-4, rel=1e-9)
    assert y[[0, 1, 3, 4, 6, 7]] == pytest.approx(y0[[0, 1, 3, 4, 6, 7]], rel=1e-14)    # only R3, R3' change
    assert abs(qp.constraint(y0, P["m"], P["Lam"], P["kappa"])) < 1e-18
    assert abs(qp.constraint(y, P["m"], P["Lam"], P["kappa"])) < 1e-18


@pytest.fixture(scope="module")
def tri():
    return {d: qp.triaxial_run(**KW, delta=d, t_back=1e-2, n=200) for d in (1e-4, 2e-4)}


def test_the_anisotropy_freezes_after_inflation_starts(tri):
    for d, run in tri.items():
        r = qp.anisotropy_ratio(run["forward"])
        assert r[0] == pytest.approx(d, rel=1e-9)
        assert r[-40:] == pytest.approx(np.full(40, r[-1]), rel=2e-3, abs=1e-12)     # frozen at the end
        assert abs(r[-1]) <= d                                                          # and no larger than it began
        assert np.all(np.abs(r) < 1.5 * d)
        # isotropy of R1/R2 survives the perturbation
        R = run["forward"]["R"]
        assert R[-1, 0] / R[-1, 1] == pytest.approx(1.0, abs=5e-3)


def test_triaxial_perturbation_makes_the_curvature_singular_toward_the_pancake(tri):
    t_ref = 1.0
    K = {}
    for d, run in tri.items():
        b = run["backward"]
        Kd = qp.pancake_kretschmann(b)
        K[d] = (b["t"], Kd)
        for tt in (1.0, 0.1, 0.03):
            i = int(np.argmin(np.abs(b["t"] - tt)))
            est = 512.0 * d * d / P["a0"] ** 4 / b["t"][i] ** 4          # K ~ 512 delta^2 / (a0^4 t^4), order of magnitude
            assert 0.3 < abs(Kd[i]) / est < 3.0
    t1, K1 = K[1e-4]
    t2, K2 = K[2e-4]
    i = int(np.argmin(np.abs(t1 - t_ref)))
    assert K2[i] / K1[i] == pytest.approx(4.0, rel=0.1)                   # K ~ delta^2
    k_far, k_near = abs(K1[int(np.argmin(np.abs(t1 - 10.0)))]), abs(K1[int(np.argmin(np.abs(t1 - 0.03)))])
    assert k_near > 1e8 * k_far                                           # grows by many orders toward t = 0
    K0 = qp.kretschmann_at_pancake(P["b0"], P["f0"], P["m"], P["Lam"], P["kappa"], P["a0"])
    assert k_near > 1e15 * K0                                             # far above the biaxial pancake value


# ---------------------------------------------------------------- winding geodesics

def test_x_geodesic_matches_the_frame_equation_and_conserves_p_x(paper_sol):
    L = 5.0
    g = qp.x_geodesic(paper_sol, L)
    pos = paper_sol["t"] > 0
    t, R1 = paper_sol["t"][pos], paper_sol["R"][pos, 0]
    # gamma from p_x = L conserved equals the frame-equation result: a1 w1 = const with a1 = R1 / 2 -> w1 = 2 L / R1
    w = np.zeros((len(t), 3))
    w[:, 0] = 2.0 * L / R1
    assert g["gamma"][pos] == pytest.approx(np.sqrt(1.0 + np.sum(w ** 2, axis=1)), rel=1e-13)
    # and the frame ODE keeps a1 w1 constant with w2 = w3 = 0 (curvature term needs w_j w_k != 0)
    a = np.abs(paper_sol["R"][pos]) / 2.0
    H = paper_sol["Rd"][pos] / paper_sol["R"][pos]
    for i in (0, 100, 200):
        dw, dt = bk.geodesic_frame_rhs(w[i], a[i], H[i], branch=+1)
        assert dw == pytest.approx([-dt * H[i, 0] * w[i, 0], 0.0, 0.0], rel=1e-12)
        # d(a1 w1)/dtau = a1 (dw1 + dt H1 w1) = 0
        assert a[i, 0] * (dw[0] + dt * H[i, 0] * w[i, 0]) == pytest.approx(0.0, abs=1e-12 * abs(a[i, 0] * w[i, 0]))


def test_winding_diverges_logarithmically_while_the_crossing_is_brief(paper_sol):
    L = 5.0
    g = qp.x_geodesic(paper_sol, L)
    t = paper_sol["t"]
    near = (np.abs(t) > 1e-3) & (np.abs(t) < 1e-1)
    # dx / d ln|t| -> 2 / a0 on both sides, signed with t
    assert np.abs(g["dxdlnt"][near]) == pytest.approx(2.0 / P["a0"], rel=1e-4)
    assert np.all(g["dxdt"] > 0.0)                                       # x increases through the pancake
    pos = np.flatnonzero(t > 0)
    # x grows like (2/a0) ln t: compare the span between two grid points
    i, j = pos[5], pos[60]
    assert g["x_rel"][j] - g["x_rel"][i] == pytest.approx((2.0 / P["a0"]) * np.log(t[j] / t[i]), rel=2e-3)
    # proper time spent close to the pancake is tiny and follows a0 t^2 / (4 |L|) per side
    assert g["tau_rel"][pos[10]] == pytest.approx(P["a0"] * (t[pos[10]] ** 2 - t[pos[0]] ** 2) / (4 * abs(L)), rel=1e-3)
    assert qp.crossing_proper_time(P["a0"], L, 1e-3) == pytest.approx(P["a0"] * 1e-6 / (2 * L), rel=1e-14)
    assert qp.x_winding(P["a0"], 1.0, 1e-6) == pytest.approx(2.0 * (2.0 / P["a0"]) * np.log(1e6), rel=1e-14)
    assert qp.x_winding(P["a0"], 1.0, 1e-12) > 2.0 * qp.x_winding(P["a0"], 1.0, 1e-6) * 0.99


def test_gamma_diverges_at_the_pancake_and_is_one_far_away(paper_sol):
    g = qp.x_geodesic(paper_sol, 5.0)
    t = paper_sol["t"]
    assert g["gamma"][np.argmin(np.abs(t))] > 1e3
    assert g["gamma"][-1] == pytest.approx(1.0, abs=1e-12)
