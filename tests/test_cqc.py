"""Tests for qgl_cqc: the 4D (E8 -> 600-cell) side of the compound of five cuboctahedra."""
import numpy as np
import pytest

import qgl_cqc as c


@pytest.fixture(scope="module")
def setup():
    return c._setup()


def test_shell_is_a_600_cell(setup):
    v = setup[0]
    assert v.shape == (120, 4)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    gram = v @ v.T
    assert np.all(np.sum(np.isclose(gram, c.qsn.TAU / 2.0, atol=1e-7), axis=1) == 12)


def test_symmetry_group_has_7200_rotations(setup):
    assert len(setup[1]) == 7200


def test_ten_partitions_into_five_24cells_split_five_and_five_by_chirality(setup):
    v, rots, parts = setup
    assert len(parts) == 10
    for p in parts:
        assert len(set().union(*p)) == 120 and all(len(x) == 24 for x in p)
    plus = sum(c.five_fold_isoclinic(v, p, 1, rots) is not None for p in parts)
    minus = sum(c.five_fold_isoclinic(v, p, -1, rots) is not None for p in parts)
    assert (plus, minus) == (5, 5)


def test_a_24cell_has_12_cuboctahedral_equators(setup):
    v, _, parts = setup
    eq = c.equators(v[sorted(parts[0][0])])
    assert len(eq) == 12 and all(len(on) == 12 for _, on in eq)


@pytest.mark.parametrize("g_chirality", [1, -1])
def test_same_side_isoclinic_rotations_give_one_cuboctahedron(g_chirality):
    _, n = c.c5c_from_4d(g_chirality, "same")
    assert n == 12


@pytest.mark.parametrize("g_chirality", [1, -1])
def test_opposite_side_isoclinic_rotations_give_the_compound_of_five_cuboctahedra(g_chirality):
    pts, n = c.c5c_from_4d(g_chirality, "opposite")
    assert n == 60
    ref = c.distance_signature(c.reference_c5c())
    assert np.allclose(c.distance_signature(pts.reshape(-1, 3)), ref, atol=1e-6)
