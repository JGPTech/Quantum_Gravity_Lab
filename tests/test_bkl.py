"""BKL / Mixmaster: the bounce map, vacuum Bianchi IX dynamics, exact curvature, and a sensor
riding through a bounce (qgl_bkl).

The algebra and dynamics tests need only numpy/scipy.  The curvature validations need sympy
(they build a brute-force Riemann tensor of the Euler-angle metric) and are skipped without it.
All tests run in a few seconds.
"""
import itertools
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import qgl_bkl as bkl
import qgl_kasner as qk
from unified_sandbox_core import Config


def exps(u):
    """Kasner exponents (neg, mid, large) from the parametrisation, valid for any u > 0 (u < 1 reorders)."""
    d = 1 + u + u * u
    return np.array([-u / d, (1 + u) / d, u * (1 + u) / d])


# ---------------------------------------------------------------- the BKL map (algebra)

@pytest.mark.parametrize("u", [3.7, 2.5, 2.0, 1.7, 1.2])
def test_landau_lifshitz_rule_is_the_u_map(u):
    p = exps(u)                                       # axes 0, 1, 2 = neg, mid, large
    q = bkl.bkl_transition(p)
    nxt = exps(u - 1.0)                               # parameter after one bounce
    # the negative exponent moves to the old middle axis, the old negative axis becomes the middle one
    assert q == pytest.approx([nxt[1], nxt[0], nxt[2]], abs=1e-12)
    if u < 2.0:                                       # new era: parameter 1/(u-1); u-1 < 1 reorders mid and large
        assert sorted(q) == pytest.approx(sorted(exps(bkl.bkl_u_map(u)[0])), abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_transition_preserves_both_kasner_conditions(seed):
    p = exps(np.random.default_rng(seed).uniform(1.05, 9.0))[np.random.default_rng(seed).permutation(3)]
    q = bkl.bkl_transition(p)
    assert q.sum() == pytest.approx(1.0, abs=1e-12)
    assert (q ** 2).sum() == pytest.approx(1.0, abs=1e-12)
    assert np.sum(q < 0) == 1


def test_transition_needs_exactly_one_negative_exponent():
    with pytest.raises(ValueError):
        bkl.bkl_transition([0.0, 0.0, 1.0])           # flat Kasner: no bounce


def test_u_map_branches():
    assert bkl.bkl_u_map(3.7) == (pytest.approx(2.7), False)
    assert bkl.bkl_u_map(1.25) == (pytest.approx(4.0), True)
    for bad in (1.0, 0.5):
        with pytest.raises(ValueError):
            bkl.bkl_u_map(bad)


def test_era_sequence_counts_eras():
    seq = bkl.era_sequence(3.7, 7)
    assert [round(s["u"], 3) for s in seq] == [3.7, 2.7, 1.7, 1.429, 2.333, 1.333, 3.0]
    assert [s["era"] for s in seq] == [0, 0, 0, 1, 2, 2, 3]


# ---------------------------------------------------------------- the equations

@pytest.mark.parametrize("seed", range(4))
def test_equations_propagate_the_hamiltonian_constraint(seed):
    """With C~ = 2 sum H_i H_j + R3 (constraint / t^2), d C~/d eta = -2 C~ along the flow: C~ = 0 stays 0."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([rng.normal(size=3) * 0.5, rng.normal(size=3) * 0.6 + 1.0])

    def c_tilde(yy):
        x, H = yy[:3], yy[3:]
        q = np.exp(2 * x)
        R3 = (2 * (q[0] * q[1] + q[1] * q[2] + q[2] * q[0]) - np.sum(q ** 2)) / (2 * np.prod(q))
        return 2 * (H[0] * H[1] + H[1] * H[2] + H[2] * H[0]) + R3

    h, d = 1e-6, bkl._rhs_eta(0.0, y)
    dC = (c_tilde(y + h * d) - c_tilde(y - h * d)) / (2 * h)
    assert dC == pytest.approx(-2.0 * c_tilde(y), rel=1e-5)


# ---------------------------------------------------------------- the numerical solution

@pytest.fixture(scope="module")
def mm():
    return bkl.Mixmaster(3.7, t0=1.0e6, t_bounce=1.0, t_end=1.0e-60)


def test_solution_starts_where_asked_and_satisfies_the_constraint(mm):
    assert mm.t0 == pytest.approx(1.0e6, rel=1e-2)
    assert mm.t_end == pytest.approx(1.0e-60, rel=0.1)
    assert np.max(np.abs(mm.constraint(np.geomspace(mm.t0, mm.t_end, 500)))) < 1e-7


def test_invalid_times_are_rejected():
    with pytest.raises(ValueError):
        bkl.Mixmaster(3.7, t0=10.0, t_bounce=20.0, t_end=1e-3)


def test_bounces_are_produced_by_the_dynamics(mm):
    bt = mm.bounce_times()
    assert len(bt) >= 3
    assert np.all(np.diff(bt) < 0)                    # found in order of decreasing t
    assert 1.0 < bt[0] < 20.0                         # first bounce near the requested t_bounce = 1


def test_measured_epochs_follow_the_bkl_map(mm):
    """Each measured Kasner epoch is the Landau-Lifshitz image of the one before it, and of the u-sequence."""
    ep = mm.epochs(n_grid=60000, threshold=2e-3)
    assert len(ep) >= 4
    predicted = np.array(qk.kasner_exponents(3.7))
    for i, e in enumerate(ep[:4]):
        assert e["v"].sum() == pytest.approx(1.0, abs=2e-3)               # Kasner conditions hold in the plateau
        assert (e["v"] ** 2).sum() == pytest.approx(1.0, abs=2e-3)
        assert e["v"] == pytest.approx(predicted, abs=2e-3)               # matches the u-sequence prediction
        if i + 1 < len(ep):
            assert ep[i + 1]["v"] == pytest.approx(bkl.bkl_transition(e["v"]), abs=3e-3)   # and the measured map
        predicted = bkl.bkl_transition(predicted)
    assert [e["neg_axis"] for e in ep[:4]] == [0, 1, 0, 1]


def test_inside_an_epoch_K_is_the_kasner_value(mm):
    for t in (1e-3, 1e-10, 1e-20):
        _, v = mm.state(t)
        assert mm.kretschmann(t)[0] * t ** 4 == pytest.approx(-16.0 * np.prod(v[0]), rel=1e-4)
        # K* ~ sqrt(curvature term) * K is still ~0.5% of K at t = 1e-3, so the magnitude equals |K| only to ~1e-5 there
        assert mm.weyl_magnitude(t)[0] == pytest.approx(abs(mm.kretschmann(t)[0]), rel=1e-4)


def test_K_changes_sign_at_a_bounce_but_the_weyl_magnitude_does_not_vanish(mm):
    assert mm.kretschmann(1e4)[0] > 0.0               # electric part dominates in a Kasner epoch
    assert mm.kretschmann(8.0)[0] < 0.0               # magnetic part wins near the bounce
    assert mm.kretschmann(1e-3)[0] > 0.0              # and electric again afterwards
    t = np.geomspace(1e4, 1e-3, 600)
    assert np.all(mm.weyl_magnitude(t) > 0.0)
    assert np.all(np.isfinite(mm.weyl_magnitude(t)))


# ---------------------------------------------------------------- exact curvature vs a brute-force Riemann tensor

@pytest.fixture(scope="module")
def brute():
    """Return f(a, ad, add) -> (K, K*) from the full Riemann tensor of the Euler-angle Bianchi IX metric."""
    sp = pytest.importorskip("sympy")
    t, th, ph, ps = sp.symbols("t th ph ps")
    A, B, C = sp.symbols("A1:4"), sp.symbols("B1:4"), sp.symbols("C1:4")
    fa = [sp.Function(f"a{i}")(t) for i in (1, 2, 3)]
    sig = [[0, sp.cos(ps), sp.sin(ps) * sp.sin(th), 0], [0, -sp.sin(ps), sp.cos(ps) * sp.sin(th), 0],
           [0, 0, sp.cos(th), 1]]
    g = sp.zeros(4, 4)
    g[0, 0] = -1
    for i in range(3):
        for m in range(4):
            for n in range(4):
                g[m, n] += fa[i] ** 2 * sig[i][m] * sig[i][n]
    X = [t, th, ph, ps]

    def to_syms(e):
        for i in range(3):
            e = e.subs(sp.Derivative(fa[i], (t, 2)), C[i]).subs(sp.Derivative(fa[i], t), B[i]).subs(fa[i], A[i])
        return e

    args = [t, th, ph, ps, *A, *B, *C]
    gf = sp.lambdify(args, g.applyfunc(to_syms), "numpy")
    dgf = sp.lambdify(args, [[[to_syms(sp.diff(g[m, n], X[l])) for n in range(4)] for m in range(4)]
                             for l in range(4)], "numpy")
    ddgf = sp.lambdify(args, [[[[to_syms(sp.diff(g[m, n], X[l], X[q])) for n in range(4)] for m in range(4)]
                               for l in range(4)] for q in range(4)], "numpy")
    eps = np.zeros((4, 4, 4, 4))
    for perm in itertools.permutations(range(4)):
        eps[perm] = np.linalg.det(np.eye(4)[list(perm)])

    def run(a, ad, add):
        # theta = pi/2, psi = 0 makes the coordinate basis line up with the frame: the metric is diagonal
        # there, so it stays well conditioned even when the scale factors differ by many orders of magnitude
        pt = (0.0, np.pi / 2, 0.0, 0.0, *a, *ad, *add)
        G = np.array(gf(*pt), float)
        Gi = np.linalg.inv(G)
        dg, ddg = np.array(dgf(*pt), float), np.array(ddgf(*pt), float)
        Gam_l, dGam_l = np.zeros((4, 4, 4)), np.zeros((4, 4, 4, 4))
        for l, n, s in itertools.product(range(4), repeat=3):
            Gam_l[l, n, s] = 0.5 * (dg[n, l, s] + dg[s, l, n] - dg[l, n, s])
            for m in range(4):
                dGam_l[m, l, n, s] = 0.5 * (ddg[m, n, l, s] + ddg[m, s, l, n] - ddg[m, l, n, s])
        Gam = np.einsum("rl,lns->rns", Gi, Gam_l)
        dGi = -np.einsum("ra,mab,bl->mrl", Gi, dg, Gi)
        dGam = np.einsum("mrl,lns->mrns", dGi, Gam_l) + np.einsum("rl,mlns->mrns", Gi, dGam_l)
        Rud = (np.einsum("mrns->rsmn", dGam) - np.einsum("nrms->rsmn", dGam)
               + np.einsum("rml,lns->rsmn", Gam, Gam) - np.einsum("rnl,lms->rsmn", Gam, Gam))
        Rd = np.einsum("ra,asmn->rsmn", G, Rud)
        Ru = np.einsum("rsmn,ra,sb,mc,nd->abcd", Rd, Gi, Gi, Gi, Gi)
        eps_up = -eps / np.sqrt(abs(np.linalg.det(G)))              # Lorentzian: eps^{0123} = -1/sqrt|g|
        Rddu = np.einsum("abcd,ce,df->abef", Rd, Gi, Gi)
        return (float(np.sum(Rd * Ru)),
                0.5 * float(np.einsum("abef,abcd,efcd->", eps_up, Rd, Rddu)))
    return run


@pytest.mark.parametrize("seed", range(3))
def test_kretschmann_formula_matches_brute_force_off_shell(brute, seed):
    rng = np.random.default_rng(seed)
    a, ad, add = rng.uniform(0.6, 1.6, 3), rng.normal(0, 0.5, 3), rng.normal(0, 0.7, 3)
    assert bkl.bianchi9_kretschmann(a, ad, add) == pytest.approx(brute(a, ad, add)[0], rel=1e-9)


@pytest.mark.parametrize("t", [30.0, 8.0, 3.0, 1.5])
def test_weyl_invariants_match_brute_force_on_the_solution(brute, mm, t):
    """K and the dual invariant K* on a real solution, with a'' taken by finite differences (not from the field equations)."""
    a, ad, _ = (v[0] for v in mm.scale_factors(t))
    h = 1e-5 * t
    add_fd = (mm.scale_factors(t + h)[1][0] - mm.scale_factors(t - h)[1][0]) / (2 * h)
    K_b, Ks_b = brute(a, ad, add_fd)
    K, Ks = (float(v[0]) for v in mm.weyl_invariants(t))
    scale = np.hypot(K, Ks)
    assert K == pytest.approx(K_b, abs=1e-4 * scale)
    assert Ks == pytest.approx(Ks_b, abs=1e-4 * scale)


# ---------------------------------------------------------------- a sensor through the bounce

@pytest.fixture(scope="module")
def run(mm):
    t_p = float(mm.bounce_times()[0])
    return t_p, bkl.mixmaster_sensor_run(mm, 8.0 * t_p, t_p / 8.0)


def test_sensor_window_and_clock(run):
    t_p, df = run
    assert df["t"].iloc[0] == pytest.approx(8.0 * t_p) and df["t"].iloc[-1] == pytest.approx(t_p / 8.0)
    assert df["tau"].iloc[0] == 0.0 and np.all(np.diff(df["tau"]) > 0.0)
    assert df["tau"].iloc[-1] == pytest.approx(8.0 * t_p - t_p / 8.0)     # comoving observer: tau = t_start - t


def test_sensor_rides_through_a_sign_change_of_K(run):
    _, df = run
    assert df["K_signed"].min() < 0.0 < df["K_signed"].max()              # the bounce is inside the window
    assert np.all(df["K"] > 0.0)                                          # the magnitude never vanishes
    assert df["C_Q"].is_monotonic_increasing and 0.0 < df["C_Q"].min() < df["C_Q"].max() < 1.0
    assert df["rho_hat"].iloc[-1] > 10.0 * df["rho_hat"].iloc[0]


def test_sensor_state_is_physical(run):
    _, df = run
    assert np.all(np.abs(df["sigma_z"]) <= 1.0 + 1e-9)
    assert np.all((df["purity"] > 0.0) & (df["purity"] <= 1.0 + 1e-9))
    assert np.all(np.isfinite(df[["n", "sigma_z", "c", "omega_c", "C_Q"]].to_numpy()))


def test_sensor_feels_the_curvature_only_through_s_rho(mm, run):
    t_p, df = run
    off = Config()
    off.s_rho, off.s_kappa = 0.0, 0.0
    quiet = bkl.mixmaster_sensor_run(mm, 8.0 * t_p, t_p / 8.0, cfg=off)
    assert np.ptp(quiet["omega_c"]) < 1e-9 * quiet["omega_c"].iloc[0]
    assert np.ptp(df["omega_c"]) > 1.0
    m = min(len(df), len(quiet))
    assert np.max(np.abs(df["n"].to_numpy()[:m] - quiet["n"].to_numpy()[:m])) > 1e-3


def test_sensor_run_does_not_modify_the_callers_config(mm):
    cfg = Config()
    g0 = cfg.g_0
    t_p = float(mm.bounce_times()[0])
    bkl.mixmaster_sensor_run(mm, 4.0 * t_p, t_p / 4.0, cfg=cfg)
    assert cfg.g_0 == g0


def test_bad_sensor_window_is_rejected(mm):
    with pytest.raises(ValueError):
        bkl.mixmaster_sensor_run(mm, 1.0, 2.0)
