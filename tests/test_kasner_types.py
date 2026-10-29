"""Singularity morphology of Kasner-type scale factors (qgl_kasner.singularity_type)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import qgl_kasner as qk


@pytest.mark.parametrize("p,expected", [
    ((1 / 3, 1 / 3, 1 / 3), "point"),
    ((-1 / 3, 2 / 3, 2 / 3), "cigar"),
    ((0.5, 0.5, 0.0), "barrel"),
    ((1.0, 0.0, 0.0), "pancake"),
    ((1.0, -0.2, -0.3), "pancake"),
    ((0.0, 0.0, 0.0), "none"),
    ((-0.5, -0.5, 1.0), "pancake"),
    ((-0.5, -0.2, 0.0), "none"),
])
def test_basic_types(p, expected):
    assert qk.singularity_type(p) == expected


def test_permutation_invariant():
    p = (-1 / 3, 2 / 3, 2 / 3)
    for perm in ((0, 1, 2), (1, 2, 0), (2, 0, 1), (1, 0, 2)):
        assert qk.singularity_type([p[i] for i in perm]) == "cigar"


@pytest.mark.parametrize("u", [1, 1.5, 2, 3.7, 10, 1e3, 1e6])
def test_vacuum_kasner_is_a_cigar_at_the_singularity(u):
    p = qk.kasner_exponents(u)
    assert qk.is_vacuum_kasner(p, 1e-9)
    assert qk.singularity_type(p) == "cigar"
    assert not qk.is_flat_kasner(p, 1e-12)


def test_flat_kasner_is_the_only_vacuum_pancake():
    p = (0.0, 0.0, 1.0)
    assert qk.is_vacuum_kasner(p) and qk.is_flat_kasner(p)
    assert qk.singularity_type(p) == "pancake"        # scale factors only; the spacetime is flat


def test_vacuum_kasner_late_end_is_pancake_but_not_singular():
    p = qk.kasner_exponents(2.0)
    assert qk.singularity_type(p, "late") == "pancake"
    assert qk.kretschmann_closed_form(p, 1e6) < 1e-20


def test_direction_flips_the_sign_of_the_exponents():
    p = (-1 / 3, 2 / 3, 2 / 3)
    assert qk.singularity_type(p, "late") == qk.singularity_type([-x for x in p], "singularity")


def test_bad_direction_rejected():
    with pytest.raises(ValueError):
        qk.singularity_type((0, 0, 1), "future")


def test_pancake_is_incompatible_with_stiff_fluid_bianchi_I():
    # Pancake needs p1 > 0 and p2, p3 <= 0.  With sum p = 1 that forces p1 >= 1 and sum p^2 >= 1, with equality
    # only for (1, 0, 0).  Vacuum has sum p^2 = 1 and a stiff fluid needs sum p^2 <= 1, so only the flat case fits.
    rng = np.random.default_rng(7)
    for _ in range(2000):
        p2, p3 = -rng.uniform(1e-3, 3.0, size=2)
        p = (1.0 - p2 - p3, p2, p3)
        assert qk.singularity_type(p) == "pancake"
        assert sum(x * x for x in p) > 1.0
