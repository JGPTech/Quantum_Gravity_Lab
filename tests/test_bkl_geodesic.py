"""Moving observers in Mixmaster (qgl_bkl.geodesic_frame_rhs / mixmaster_geodesic / mixmaster_sensor_run(w0=...)).

The frame equations are checked against an independent integration of the geodesic equation with
coordinate Christoffel symbols, using explicit (off-shell) scale factors a_i(t) so the check is a
statement about the geodesic equation alone, not about the vacuum solution.
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

sp = pytest.importorskip("sympy")
from scipy.integrate import solve_ivp

import qgl_bkl as bk


# ---------------------------------------------------------------- coordinate reference

@pytest.fixture(scope="module")
def reference():
    t, th, ph, ps = sp.symbols("t th ph ps")
    a_expr = [1.3 + 0.2 * t, 0.9 + 0.1 * t ** 2, 1.1 + 0.15 * t - 0.02 * t ** 2]
    sig = [[0, sp.cos(ps), sp.sin(ps) * sp.sin(th), 0],
           [0, -sp.sin(ps), sp.cos(ps) * sp.sin(th), 0],
           [0, 0, sp.cos(th), 1]]
    g = sp.zeros(4, 4)
    g[0, 0] = -1
    for i in range(3):
        for m in range(4):
            for n in range(4):
                g[m, n] += a_expr[i] ** 2 * sig[i][m] * sig[i][n]
    X = [t, th, ph, ps]
    return dict(
        gf=sp.lambdify(X, g, "numpy"),
        dgf=sp.lambdify(X, [[[sp.diff(g[m, n], X[l]) for n in range(4)] for m in range(4)] for l in range(4)], "numpy"),
        sigf=sp.lambdify(X, sp.Matrix(sig), "numpy"),
        af=sp.lambdify(t, a_expr, "numpy"),
        adf=sp.lambdify(t, [sp.diff(e, t) for e in a_expr], "numpy"),
    )


def coord_geodesic(ref, x0, u0, T):
    def rhs(tau, y):
        x, u = y[:4], y[4:]
        Gi = np.linalg.inv(np.array(ref["gf"](*x), float))
        dg = np.array(ref["dgf"](*x), float)            # dg[l, m, n] = d_l g_mn
        Gam = 0.5 * (np.einsum("ml,alb->mab", Gi, dg) + np.einsum("ml,bla->mab", Gi, dg)
                     - np.einsum("ml,lab->mab", Gi, dg))
        return np.concatenate([u, -np.einsum("mab,a,b->m", Gam, u, u)])
    return solve_ivp(rhs, [0, T], np.concatenate([x0, u0]), method="DOP853",
                     rtol=1e-12, atol=1e-13, dense_output=True)


@pytest.mark.parametrize("branch", [+1, -1])
def test_frame_equations_match_coordinate_geodesic(reference, branch):
    ref = reference
    t0, ang = 0.8, (1.1, 0.3, 0.7)
    w0 = np.array([0.4, -0.3, 0.5])
    gam0 = np.sqrt(1 + w0 @ w0)
    x0 = np.array([t0, *ang])
    S = np.array(ref["sigf"](*x0), float)[:, 1:]
    u_sp = np.linalg.solve(S, w0 / np.array(ref["af"](t0), float))
    u0 = np.concatenate([[branch * gam0], u_sp])
    assert u0 @ np.array(ref["gf"](*x0), float) @ u0 == pytest.approx(-1.0, abs=1e-12)

    T = 0.5
    sc = coord_geodesic(ref, x0, u0, T)

    def frame_rhs(tau, y):
        a = np.array(ref["af"](y[3]), float)
        H = np.array(ref["adf"](y[3]), float) / a
        dw, dt = bk.geodesic_frame_rhs(y[:3], a, H, branch=branch)
        return np.concatenate([dw, [dt]])

    sf = solve_ivp(frame_rhs, [0, T], np.concatenate([w0, [t0]]), method="DOP853",
                   rtol=1e-12, atol=1e-13, dense_output=True)
    for tau in np.linspace(0.1, T, 5):
        yc = sc.sol(tau)
        Sc = np.array(ref["sigf"](*yc[:4]), float)[:, 1:]
        wc = np.array(ref["af"](yc[0]), float) * (Sc @ yc[5:])
        yf = sf.sol(tau)
        assert yf[:3] == pytest.approx(wc, abs=1e-9)
        assert yf[3] == pytest.approx(yc[0], abs=1e-9)


# ---------------------------------------------------------------- pure RHS properties

def test_isotropic_case_conserves_covariant_momentum():
    # equal scale factors: curvature term vanishes, w_i evolves only through -(dt/dtau) H w_i, i.e. a w = const
    a = np.array([2.0, 2.0, 2.0])
    H = np.array([0.3, 0.3, 0.3])
    w = np.array([0.2, -0.4, 0.1])
    dw, dt = bk.geodesic_frame_rhs(w, a, H, branch=+1)
    gamma = np.sqrt(1 + w @ w)
    assert dt == pytest.approx(gamma)
    assert dw == pytest.approx(-gamma * H * w)
    # d(a w)/dtau = a (dw + dt H w) = 0
    assert a * (dw + dt * H * w) == pytest.approx(np.zeros(3), abs=1e-15)


def test_comoving_observer_stays_at_rest():
    dw, dt = bk.geodesic_frame_rhs(np.zeros(3), np.array([1.0, 2.0, 3.0]), np.array([.1, .2, .3]), branch=-1)
    assert dw == pytest.approx(np.zeros(3))
    assert dt == pytest.approx(-1.0)


# ---------------------------------------------------------------- Mixmaster geodesics

@pytest.fixture(scope="module")
def mm():
    return bk.Mixmaster(3.7, 1e6, 1.0, 1e-60)


def test_comoving_geodesic_is_trivial(mm):
    t = np.geomspace(40.0, 1e-3, 30)
    g = bk.mixmaster_geodesic(mm, (0, 0, 0), t)
    assert g["tau"] == pytest.approx(40.0 - t, rel=1e-8)
    assert np.all(g["speed"] == 0.0) and np.all(g["gamma"] == 1.0)


def test_moving_observer_ages_less_and_stays_subluminal(mm):
    t = np.geomspace(40.0, 1e-3, 60)
    g = bk.mixmaster_geodesic(mm, (0.5, -0.3, 0.8), t)
    assert np.all(np.diff(g["tau"]) > 0.0)
    assert g["tau"][-1] < 40.0 - t[-1]
    assert np.all((g["speed"] > 0.0) & (g["speed"] < 1.0))
    assert g["gamma"][-1] > g["gamma"][0]            # boosted by the collapsing axes
    assert g["gamma"] == pytest.approx(np.sqrt(1 + np.sum(g["w"] ** 2, axis=1)))


def test_covariant_momentum_drifts_only_through_the_curvature_term(mm):
    # P_i = a_i w_i is conserved except for the Euler-type curvature term of the SU(2) slices, which is small
    # early in the first era and acts over a window; the drift therefore shrinks with the window.
    def drift(t_hi, factor):
        t = np.geomspace(t_hi, t_hi / factor, 20)
        g = bk.mixmaster_geodesic(mm, (0.5, -0.3, 0.8), t)
        P = np.exp(mm.state(t)[0]) * g["w"]
        return np.max(np.abs(P / P[0] - 1.0))
    short, long_ = drift(9e5, 1.2), drift(9e5, 4.0)
    assert short < 5e-3
    assert short < long_


def test_geodesic_input_validation(mm):
    with pytest.raises(ValueError):
        bk.mixmaster_geodesic(mm, (0, 0, 0), np.array([1.0, 2.0, 3.0]))   # ascending
    with pytest.raises(ValueError):
        bk.mixmaster_geodesic(mm, (0, 0, 0), np.array([1.0]))


# ---------------------------------------------------------------- sensor

def test_sensor_run_moving_vs_comoving(mm):
    tp = float(mm.bounce_times()[0])
    rest = bk.mixmaster_sensor_run(mm, 8 * tp, tp / 8)
    mov = bk.mixmaster_sensor_run(mm, 8 * tp, tp / 8, w0=(0.5, -0.3, 0.8))
    assert np.all(rest["speed"] == 0.0) and "gamma" not in rest.columns
    assert {"gamma", "w1", "w2", "w3"} <= set(mov.columns)
    assert mov["t"].to_numpy() == pytest.approx(rest["t"].to_numpy())
    assert mov["K"].to_numpy() == pytest.approx(rest["K"].to_numpy())      # K depends only on t
    assert mov["tau"].iloc[-1] < rest["tau"].iloc[-1]
    assert np.all(mov["speed"] > 0.0)
    assert np.max(np.abs(mov["n"].to_numpy() - rest["n"].to_numpy())) > 1e-3
    assert np.all(np.abs(mov["sigma_z"]) <= 1.0 + 1e-9)
