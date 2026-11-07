"""Tests for qgl_e8: E8 roots, the Coxeter element, and the H4 projection onto two golden-ratio 600-cell shells."""
import numpy as np

import qgl_e8 as e8


def test_roots_count_and_norm():
    r = e8.e8_roots()
    assert r.shape == (240, 8)
    assert np.allclose((r * r).sum(axis=1), 2.0)


def test_roots_closed_under_simple_reflections():
    r = e8.e8_roots()
    key = {tuple(np.round(x, 9)) for x in r}
    for alpha in e8.simple_roots():
        img = r @ e8.reflection(alpha).T
        assert {tuple(np.round(x, 9)) for x in img} == key


def test_simple_roots_are_roots_with_e8_cartan_determinant():
    a = e8.simple_roots()
    key = {tuple(np.round(x, 9)) for x in e8.e8_roots()}
    assert all(tuple(np.round(x, 9)) in key for x in a)
    assert np.isclose(np.linalg.det(a @ a.T), 1.0)       # E8 is unimodular


def test_coxeter_element_has_order_30():
    c = e8.coxeter_element()
    p = np.linalg.matrix_power(c, 30)
    assert np.allclose(p, np.eye(8), atol=1e-9)
    for k in (1, 2, 3, 5, 6, 10, 15):
        assert not np.allclose(np.linalg.matrix_power(c, k), np.eye(8), atol=1e-6)


def test_projection_rows_orthonormal_and_mutually_orthogonal():
    pp, pq = e8.h4_projection()
    assert np.allclose(pp @ pp.T, np.eye(4), atol=1e-9)
    assert np.allclose(pq @ pq.T, np.eye(4), atol=1e-9)
    assert np.allclose(pp @ pq.T, 0.0, atol=1e-9)


def test_roots_project_to_two_golden_shells_of_120():
    r = e8.e8_roots()
    pp, pq = e8.h4_projection()
    for p in (pp, pq):
        rad = np.linalg.norm(r @ p.T, axis=1)
        vals = np.unique(np.round(rad, 8))
        assert len(vals) == 2
        assert np.isclose(vals[1] / vals[0], e8.PHI, atol=1e-7)
        assert sorted(np.sum(np.isclose(rad, v, atol=1e-7)) for v in vals) == [120, 120]


def test_outer_shell_is_a_600_cell():
    r = e8.e8_roots()
    pp, _ = e8.h4_projection()
    x = r @ pp.T
    rad = np.linalg.norm(x, axis=1)
    shell = x[np.isclose(rad, rad.max(), atol=1e-7)]
    shell = shell / np.linalg.norm(shell[0])
    gram = shell @ shell.T
    nbrs = np.sum(np.isclose(gram, e8.PHI / 2.0, atol=1e-7), axis=1)
    assert np.all(nbrs == 12)                              # 600-cell: 12 neighbours per vertex
    assert np.sum(np.isclose(gram, -1.0, atol=1e-7)) == 120   # antipodal pairs


def test_model_set_window_and_spacing():
    pts = e8.model_set(n=1, radius=1.0)
    assert pts.shape[1] == 4 and len(pts) > 1
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    d[np.diag_indices(len(pts))] = np.inf
    assert d.min() > 1e-6                                   # points are distinct and discrete
