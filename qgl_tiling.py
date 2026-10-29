#!/usr/bin/env python3
"""
qgl_tiling.py -- cell-level data of the Chair44 tiling for QGL sensors.

The Chair44 (Tsiokos) tiles R^3 by a seven-cube chair whose 24 unit faces carry eight small pyramids of signed
heights.  The substitution has inflation 2, so the cell substitution on the 24 x 7 = 168 labelled unit cells
(label = (frame of the chair, which of its seven cells)) gives, by iteration, a cube of cells sigma^n(j) of side 2^n.
Every such cube is a patch of a Chair44 tiling.  This module builds those cubes and two per-cell observables:

    pyramid   the sum of the signed heights of the pyramids on the tile-surface faces of that cell
              (a function of which of the seven cells it is; the 192 heights sum to zero, so its mean is 0 exactly)
    notch     the direction d = g(1,1,1) of the missing corner of the chair that owns the cell, in {+-1}^3
              (the orientation vector of the paper; the notch vectors of a depth-n supertile sum to 2^n (1,1,1))

It needs the Spectre repository (symchair44.py, chair44_twoadic.py, chair44_hull.py), found through the environment
variable SPECTRE_DIR or at ../spectre.  `available()` says whether it was found; the tests skip if not.

What is checked here (tests/test_tiling.py): the label cubes agree cell for cell with the tile-level data of
chair44_slice (notch directions of every cell of a depth-3 supertile), and the notch sum of a supertile is 7 * 2^n
(1,1,1) over cells.  The coupling of any of this to a sensor is a modelling choice, as everywhere in QGL.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [os.environ.get("SPECTRE_DIR", ""), os.path.join(_HERE, "..", "spectre"), "/home/claude/spectre"]
SPECTRE_DIR: Optional[str] = next((p for p in _CANDIDATES if p and os.path.isfile(os.path.join(p, "symchair44.py"))),
                                  None)

_S = None          # symchair44
_STATE: Dict[str, object] = {}


def available() -> bool:
    return SPECTRE_DIR is not None


def _load():
    global _S
    if _S is None:
        if SPECTRE_DIR is None:
            raise RuntimeError("the Spectre repository was not found; set SPECTRE_DIR")
        if SPECTRE_DIR not in sys.path:
            sys.path.insert(0, SPECTRE_DIR)
        import symchair44 as S
        import chair44_twoadic as TA
        _S = S
        labels, sigma = TA.labels_and_sigma()
        notch = np.array([S.frame_apply(f, (1, 1, 1)) for f, _ in labels], dtype=np.int8)       # (168, 3)
        _STATE.update(labels=labels, sigma=np.array(sigma, dtype=np.int16), notch=notch,
                      index={l: i for i, l in enumerate(labels)})
    return _S


# --------------------------------------------------------------------------
#  Observables
# --------------------------------------------------------------------------

def pyramid_per_native_cell() -> np.ndarray:
    """(7,) sum of signed pyramid heights over the tile-surface faces of each native cell (S.CELLS order)."""
    S = _load()
    h = np.zeros(len(S.CELLS))
    lab = S.label_table(S.CHILDREN)
    for pi, (ax, k) in enumerate(S.PANELS):
        sd = S.PANEL_SIGN[pi]
        cell = [(k[i] - 1 - (sd if i == ax else 0)) // 2 for i in range(3)]
        h[S.CELLS.index(tuple(cell))] += sum(lab[pi * 8 + j] for j in range(8))
    return h


def observables() -> Dict[str, np.ndarray]:
    """Per-label arrays (168,): 'pyramid', 'notch_x', 'notch_y', 'notch_z', each as float."""
    if "obs" not in _STATE:
        _load()
        h = pyramid_per_native_cell()
        native = np.array([k for _, k in _STATE["labels"]])
        nt = _STATE["notch"].astype(float)
        _STATE["obs"] = dict(pyramid=h[native], notch_x=nt[:, 0], notch_y=nt[:, 1], notch_z=nt[:, 2])
    return _STATE["obs"]


# --------------------------------------------------------------------------
#  Label cubes
# --------------------------------------------------------------------------

def _inflate(A: np.ndarray) -> np.ndarray:
    """(N,N,N) labels -> (2N,2N,2N): every cell replaced by its 2x2x2 children (index b = a0*4 + a1*2 + a2)."""
    sg = _STATE["sigma"]
    N = A.shape[0]
    out = np.empty((2 * N,) * 3, dtype=np.int16)
    for b in range(8):
        out[(b >> 2) & 1::2, (b >> 1) & 1::2, b & 1::2] = sg[A, b]
    return out


def label_cube(j: int, depth: int) -> np.ndarray:
    """sigma^depth of the cell label j: an int16 cube of side 2^depth, a patch of a Chair44 tiling."""
    _load()
    A = np.array([[[j]]], dtype=np.int16)
    for _ in range(depth):
        A = _inflate(A)
    return A


def supertile_labels(depth: int) -> np.ndarray:
    """Labels of the depth-n supertile of the root chair on a (2^(n+1))^3 grid; -1 where the notch octant is empty."""
    S = _load()
    root = S.IDENTITY
    side = 2 ** (depth + 1)
    out = -np.ones((side,) * 3, dtype=np.int16)
    for k, c in enumerate(S.CELLS):
        j = _STATE["index"][(root, k)]
        cube = label_cube(j, depth)
        x0 = [ci * 2 ** depth for ci in c]
        out[x0[0]:x0[0] + 2 ** depth, x0[1]:x0[1] + 2 ** depth, x0[2]:x0[2] + 2 ** depth] = cube
    return out


def field(cube: np.ndarray, name: str) -> np.ndarray:
    """The observable `name` evaluated on a label cube (float32; NaN where the label is -1)."""
    obs = observables()[name]
    out = np.full(cube.shape, np.nan, dtype=np.float32)
    ok = cube >= 0
    out[ok] = obs[cube[ok]]
    return out


def seed_labels() -> list:
    """The 168 label indices (all are available as starting cells)."""
    _load()
    return list(range(len(_STATE["labels"])))
