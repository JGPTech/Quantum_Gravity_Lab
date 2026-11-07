"""Tests for qgl_qsn: icosahedral frame, tetragrids, Fibonacci spacing, the 20G cluster and the icosagrid."""
import itertools

import numpy as np

import qgl_qsn as q


def test_golden_rotation_value():
    assert np.isclose(q.GOLDEN_ROTATION_DEG, 15.5225, atol=1e-4)      # the 15.522 degrees quoted by QGR


def test_icosahedral_frame_counts():
    assert len(q.icosahedral_axes()) == 10
    assert len(q.dodecahedron_vertices()) == 20
    assert len(q.inscribed_tetrahedra()) == 10


def test_five_tetrahedra_partition_the_dodecahedron_vertices():
    t = q.five_tetrahedra()
    assert t.shape == (5, 4, 3)
    keys = {tuple(np.round(v, 6)) for v in t.reshape(-1, 3)}
    assert len(keys) == 20
    for tet in t:
        assert np.allclose(tet.sum(axis=0), 0.0, atol=1e-9)          # regular tetrahedron centred at the origin
        g = tet @ tet.T
        assert np.allclose(g[~np.eye(4, dtype=bool)], -1.0 / 3.0)
    lines = {tuple(np.round(v * np.sign(next(x for x in v if abs(x) > 1e-9)), 6)) for v in t.reshape(-1, 3)}
    assert len(lines) == 10                                           # five tetragrids share the ten 3-fold axes


def test_tetragrids_are_related_by_72_degrees_and_by_golden_rotation_modulo_cube_symmetry():
    t = q.five_tetrahedra()
    angs, mod_cube = set(), set()
    for i, j in itertools.combinations(range(5), 2):
        angs.add(round(q.min_relative_angle_deg(t[i], t[j]), 6))
        rots = q.relative_rotations(t[i], t[j]) + q.relative_rotations(t[i], -t[j])
        mod_cube.add(min(q.rotation_angle_deg(r) for r in rots))
    assert angs == {72.0}
    # numerical identity: the smallest relative rotation modulo the cube group is 60 degrees minus the golden rotation
    assert all(np.isclose(60.0 - a, q.GOLDEN_ROTATION_DEG, atol=1e-9) for a in mod_cube)


def test_fibonacci_chain_is_symmetric_with_two_gaps_in_golden_ratio():
    fo = q.fibonacci_offsets(12)
    assert all(abs(fo[n] + fo[-n]) < 1e-12 for n in range(13))
    assert np.isclose(fo[1], q.TAU)                                    # first gap from the origin is the long one
    gaps = np.diff([fo[n] for n in range(-12, 13)])
    assert set(np.round(gaps, 9)) == {1.0, round(q.TAU, 9)}


def test_fibonacci_chain_is_a_factor_of_the_substitution_word_L_to_LS_S_to_L():
    # The chain is symmetric about the origin (two long gaps meet there), so its right half need not be a prefix of the
    # one-sided substitution word; it is a Sturmian word with the same language, i.e. a factor of it.
    fo = q.fibonacci_offsets(40)
    gaps = np.diff([fo[n] for n in range(-40, 41)])
    word = "".join("L" if g > 1.2 else "S" for g in gaps)
    assert word[39:41] == "LL"
    w = "L"
    for _ in range(14):
        w = "".join("LS" if c == "L" else "L" for c in w)
    assert word in w
    assert abs(word.count("L") / len(word) - 1.0 / q.TAU) < 0.03


def test_20g_has_61_distinct_vertices():
    pts = q.cluster_20g()
    assert len(pts) == 61
    assert len({tuple(np.round(p, 6)) for p in pts}) == 61


def test_uniform_tetragrid_is_fcc():
    t = q.five_tetrahedra()[0]
    p3 = q.grid_vertices(t, q.uniform_offsets(6), 3.0, min_planes=4)
    p4 = q.grid_vertices(t, q.uniform_offsets(6), 3.0, min_planes=3)
    assert len(p3) == len(p4) > 20                                      # every triple point carries the 4th plane
    d = np.linalg.norm(p3[:, None] - p3[None, :], axis=2)
    d[np.diag_indices(len(p3))] = np.inf
    nn = q.core_scale(q.uniform_offsets(6))
    assert np.isclose(d.min(), nn, atol=1e-6)                           # FCC nearest-neighbour distance


def test_fibonacci_grid_is_sparser_when_all_four_planes_must_meet():
    fo = q.fibonacci_offsets(10)
    all3 = q.icosagrid_vertices(fo, 6.0, 3)
    all4 = q.icosagrid_vertices(fo, 6.0, 4)
    assert 0 < len(all4) < len(all3)


def test_20g_sits_inside_the_fibonacci_icosagrid_at_the_right_scale_only():
    fo = q.fibonacci_offsets(10)
    pts = q.icosagrid_vertices(fo, 6.0, 4)
    lam = q.core_scale(fo)
    assert q.contains(pts, q.cluster_20g(lam)) == (61, 61)
    # negative controls: the same cluster at unrelated scales, and one step down in the golden inflation, is not contained
    for wrong in (lam / q.TAU, 0.9 * lam, 1.1 * lam, 2 ** 0.5 * lam):
        found, total = q.contains(pts, q.cluster_20g(wrong))
        assert found < total
    # observation, not a claim from the source: the grid is self-similar under scaling by tau, so inflated copies fit too
    for up in (q.TAU, q.TAU ** 2):
        assert q.contains(pts, q.cluster_20g(lam * up)) == (61, 61)


# ---------------------------------------------------------------- tetrahedral cells and the edge graph
import pytest


@pytest.fixture(scope="module")
def fib_graph():
    fo = q.fibonacci_offsets(12)
    verts, edges, cells, owner = q.icosagrid_graph(fo, 12.0)
    return fo, verts, edges, cells


def test_uniform_tetragrid_cells_are_the_fcc_tetrahedra():
    uo = q.uniform_offsets(6)
    cells = q.tetra_cells(q.five_tetrahedra()[0], uo, 3.0)
    assert len(cells) > 50
    lengths = {round(float(np.linalg.norm(c[a] - c[b])), 6) for c in cells for a in range(4) for b in range(a)}
    assert lengths == {round(q.core_scale(uo), 6)}                       # one size: the FCC nearest-neighbour distance


def test_fibonacci_tetrahedra_come_in_sizes_related_by_powers_of_tau(fib_graph):
    fo, _, _, cells = fib_graph
    lengths = sorted({round(float(np.linalg.norm(c[0] - c[1])), 6) for c in cells})
    assert len(lengths) == 4
    assert np.allclose(np.array(lengths[1:]) / np.array(lengths[:-1]), q.TAU, atol=1e-5)
    assert np.isclose(lengths[-1], q.core_scale(fo), atol=1e-6)         # largest cells: edge = the 20G cuboctahedron radius


def test_icosagrid_graph_has_30_edge_directions_and_a_60_valent_origin(fib_graph):
    fo, verts, edges, cells = fib_graph
    stats = q.graph_statistics(verts, edges, 6.0)
    assert stats["n_directions"] == 30                                   # 5 tetragrids x 6 edge directions
    origin = [i for i, p in enumerate(verts) if np.linalg.norm(p) < 1e-9][0]
    assert sum(1 for a, b in edges if origin in (a, b)) == 60


def test_largest_cell_graph_has_degrees_3_to_60_as_reported_for_the_icosagrid(fib_graph):
    fo, _, _, cells = fib_graph
    vb, eb = q.graph_from_cells(cells, q.core_scale(fo))
    stats = q.graph_statistics(vb, eb, 6.0)
    assert (stats["deg_min"], stats["deg_max"]) == (3, 60)
    assert stats["n_directions"] == 30


def test_all_sizes_together_have_minimum_degree_6_not_3(fib_graph):
    # Recorded difference: the paper's minimum degree of 3 is reproduced only by the largest-cell subgraph.
    _, verts, edges, _ = fib_graph
    stats = q.graph_statistics(verts, edges, 6.0)
    assert (stats["deg_min"], stats["deg_max"]) == (6, 60)
