"""The S^3 identifications behind qgl_horizon and qgl_pancake, checked against the geometry of the Euler-angle S^3.

The pancake modules treat the spatial slice of biaxial Bianchi IX as S^3 with the metric

    (1/4) [ R_1^2 (d psi + cos(th) d phi)^2 + R_2^2 (d th^2 + sin^2(th) d phi^2) ] ,   R_2 = R_3 ,

where psi is the Euler angle along the Hopf fibre.  Several statements in the code rest on that identification and carried
a "not checked" note: FIBRE_PERIOD = 4 pi, the fibre circumference 2 pi R_1, the base null angle int 2 dt / R_2 reaching pi
at antipodal points, and the diagonal Ricci components of the slice (qgl_bkl.ricci3).  Here each is checked from the
embedding S^3 in C^2, z1 = cos(th/2) e^{i(psi+phi)/2}, z2 = sin(th/2) e^{i(psi-phi)/2}, with a Hopf map and a Milnor
Ricci formula written independently of the modules under test.

What this does NOT settle: whether the slices of the paper (arXiv:0809.4335) are S^3 and not the quotient SO(3) = RP^3
(psi has period 2 pi there, and the winding counts would double).  It checks the standard S^3, which is what the code
assumes.
"""
import math

import numpy as np
import pytest

import qgl_horizon as hz
import qgl_pancake as qp


def euler_point(th, ph, ps):
    """The point of the unit S^3 in R^4 with Euler angles (theta, phi, psi)."""
    z1 = math.cos(th / 2) * np.exp(0.5j * (ps + ph))
    z2 = math.sin(th / 2) * np.exp(0.5j * (ps - ph))
    return np.array([z1.real, z1.imag, z2.real, z2.imag])


def hopf_base(p):
    """Hopf map of a point of R^4 = C^2: (2 Re z1 conj z2, 2 Im z1 conj z2, |z1|^2 - |z2|^2)."""
    z1, z2 = complex(p[0], p[1]), complex(p[2], p[3])
    w = z1 * np.conj(z2)
    return np.array([2 * w.real, 2 * w.imag, abs(z1) ** 2 - abs(z2) ** 2])


def milnor_ricci(R):
    """Ricci curvatures (orthonormal frame) of the left-invariant metric (1/4) sum R_i^2 omega_i^2 on S^3 (unit S^3 at R_i = 1).

    Milnor 1976: with [e_j, e_k] = l_i e_i in an orthonormal frame, Ric(e_i) = 2 mu_j mu_k, mu_i = (l1 + l2 + l3)/2 - l_i;
    for this metric l_i = 2 R_i / (R_j R_k)."""
    R1, R2, R3 = R
    l = (2 * R1 / (R2 * R3), 2 * R2 / (R3 * R1), 2 * R3 / (R1 * R2))
    h = sum(l) / 2.0
    m = [h - x for x in l]
    return np.array([2 * m[1] * m[2], 2 * m[2] * m[0], 2 * m[0] * m[1]])


def test_euler_angles_pull_back_the_round_unit_sphere():
    """g = (1/4)(d th^2 + d phi^2 + d psi^2 + 2 cos th d phi d psi) in the order (th, phi, psi): the unit S^3."""
    rng = np.random.default_rng(1)
    eps = 1e-6
    for _ in range(10):
        c = np.array([rng.uniform(0.3, 2.8), rng.uniform(0, 6.28), rng.uniform(0, 12.5)])
        J = np.stack([(euler_point(*(c + eps * np.eye(3)[i])) - euler_point(*(c - eps * np.eye(3)[i]))) / (2 * eps)
                      for i in range(3)], axis=1)
        pred = 0.25 * np.array([[1, 0, 0], [0, 1, math.cos(c[0])], [0, math.cos(c[0]), 1]])
        assert np.abs(J.T @ J - pred).max() < 1e-8


def test_psi_orbits_are_hopf_fibres():
    th, ph = 1.1, 0.7
    ref = hopf_base(euler_point(th, ph, 0.0))
    for ps in np.linspace(0, 4 * math.pi, 41):
        assert np.abs(hopf_base(euler_point(th, ph, ps)) - ref).max() < 1e-12


def test_fibre_period_is_four_pi_on_s3():
    """The point first returns at psi = 4 pi; psi = 2 pi is the antipode (on SO(3) the period would be 2 pi)."""
    th, ph = 1.1, 0.7
    p0 = euler_point(th, ph, 0.0)
    assert np.allclose(euler_point(th, ph, 2 * math.pi), -p0, atol=1e-12)
    ps = np.linspace(0.1, 4 * math.pi + 1.0, 20001)          # start away from psi = 0, where the point is trivially close
    d = np.array([np.linalg.norm(euler_point(th, ph, x) - p0) for x in ps])
    assert np.all(d[ps < 4 * math.pi - 0.05] > 1e-2)         # no earlier return, in particular none at 2 pi
    first = ps[np.argmax(d < 1e-3)]
    assert first == pytest.approx(4 * math.pi, abs=5e-3)
    assert hz.FIBRE_PERIOD == pytest.approx(4 * math.pi, rel=1e-15)


def test_fibre_circumference_is_two_pi_r1():
    """|d/d psi| = 1/2 on the unit S^3, so a fibre has length FIBRE_PERIOD / 2 = 2 pi (a great circle); scaled by R_1 it
    is 2 pi R_1, which is what qgl_horizon reports as the fibre circumference."""
    th, ph = 1.1, 0.7
    pts = np.array([euler_point(th, ph, x) for x in np.linspace(0, hz.FIBRE_PERIOD, 20001)])
    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    assert length == pytest.approx(2 * math.pi, rel=1e-8)
    assert 0.5 * hz.FIBRE_PERIOD == pytest.approx(length, rel=1e-8)
    sol = qp.pancake_solution(1.0, 3.0, 2.0, 0.5, t_max=2.0, n=100)
    h = hz.pancake_horizons(sol)
    R1 = sol["R"][sol["t"] > 0, 0]
    assert np.allclose(h["fibre_circumference"], 0.5 * R1 * hz.FIBRE_PERIOD, rtol=1e-14)
    assert np.allclose(h["fibre_turns"] * hz.FIBRE_PERIOD, h["fibre"], rtol=1e-14)


def test_base_is_a_sphere_of_radius_half_r2_with_theta_its_polar_angle():
    """theta is the polar angle of the Hopf base point, so antipodes differ by pi, and the base metric is
    (R_2/2)^2 (d th^2 + sin^2 th d phi^2): a null ray across the base has d th / dt = 2 / R_2."""
    for th, ph in [(0.4, 0.9), (1.3, 4.0), (2.6, 2.2)]:
        b = hopf_base(euler_point(th, ph, 1.7))
        assert np.allclose(b, [math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)], atol=1e-12)
    south, north = hopf_base(euler_point(math.pi, 0.0, 0.3)), hopf_base(euler_point(0.0, 0.0, 0.3))
    assert np.allclose(south, [0, 0, -1], atol=1e-12) and np.allclose(north, [0, 0, 1], atol=1e-12)
    # g_thth = 1/4 (previous test), so the base metric (R_2^2/4) d th^2 gives d th / dt = 2 / R_2 for a null ray
    eps = 1e-6
    dp = (euler_point(1.0 + eps, 0.3, 0.2) - euler_point(1.0 - eps, 0.3, 0.2)) / (2 * eps)
    assert np.linalg.norm(dp) == pytest.approx(0.5, rel=1e-8)


def test_slice_ricci_is_milnors_formula():
    """qgl_bkl.ricci3 (through qgl_pancake.slice_ricci, R = 2a) equals Milnor's Ricci curvature, unit S^3 at R_i = 1."""
    rng = np.random.default_rng(2)
    for _ in range(50):
        R = rng.uniform(0.2, 3.0, size=3)
        assert np.allclose(qp.slice_ricci(R), milnor_ricci(R), rtol=1e-12)
    assert np.allclose(qp.slice_ricci(np.ones(3)), 2.0)


@pytest.mark.parametrize("s", [0.05, 0.3, 1.0, 1.7])
def test_biaxial_slice_is_a_berger_sphere(s):
    """R = (s, 1, 1): Ricci (2 s^2, 4 - 2 s^2, 4 - 2 s^2) and scalar 8 - 2 s^2 (Berger sphere, fibre scaled by s)."""
    r = qp.slice_ricci(np.array([s, 1.0, 1.0]))
    assert np.allclose(r, [2 * s * s, 4 - 2 * s * s, 4 - 2 * s * s], rtol=1e-12)
    assert r.sum() == pytest.approx(8 - 2 * s * s, rel=1e-12)


@pytest.mark.parametrize("b0", [1.0, 3.0, 18000.0])
def test_collapsing_fibre_keeps_the_slice_curvature_finite(b0):
    """R = (1e-6 b0, b0, b0): scalar curvature 8 / b0^2 and no Ricci curvature along the fibre.  b0 = 18000 is the paper's
    value; the factored form of ricci3 is what keeps this accurate."""
    r = qp.slice_ricci(np.array([1e-6 * b0, b0, b0]))
    assert r.sum() * b0 ** 2 == pytest.approx(8.0, rel=1e-9)
    assert abs(r[0]) * b0 ** 2 < 1e-9
