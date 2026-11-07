#!/usr/bin/env python3
"""
qgl_e8.py -- E8 roots, the H4 (Coxeter-plane) projection, and a cut-and-project model set.

Construction (all numerical, nothing hard-coded beyond the E8 simple roots):

  1. e8_roots()          the 240 roots of E8: 112 of the form (+-1, +-1, 0^6) up to permutation and 128 of the form
                         (+-1/2)^8 with an even number of minus signs.
  2. coxeter_element()   product of the 8 simple reflections (Bourbaki simple roots).  It has order 30 and eigenvalue
                         angles 2*pi*m/30 for the exponents m = 1, 7, 11, 13, 17, 19, 23, 29.
  3. h4_projection()     the two 4D invariant subspaces of that element: m in {1, 11, 19, 29} (angles 12 and 132 deg)
                         and m in {7, 13, 17, 23} (angles 84 and 156 deg).  Each is returned as a 4x8 matrix with
                         orthonormal rows.  The two are Galois conjugates (sqrt5 -> -sqrt5), so they play the roles of
                         physical and internal space in a cut-and-project scheme.
  4. model_set()         lattice points y = sum n_i alpha_i with |n_i| <= n whose internal image lies in a ball of
                         radius R; returns their physical images.

What is checked (tests/test_e8.py): 240 roots of norm^2 2, closure under the simple reflections, Coxeter element of
order 30, and that the 240 roots project onto two shells of 120 points each whose radii are in the golden ratio, with
the 120-point shell having the 600-cell neighbour structure (12 neighbours at cos = phi/2).

What is NOT claimed:
  * model_set() uses a spherical window.  That gives a quasiperiodic point set, but it is not the exact window used
    for the Elser-Sloane quasicrystal in the literature (which I could not retrieve), so it should not be called that.
  * Nothing here builds the 3D Compound Quasicrystal, the 15.522 degree tetragrid rotations or the QSN of Quantum
    Gravity Research.  See QUANTGRAV.md.
"""
from __future__ import annotations

import itertools
from typing import Tuple

import numpy as np

PHI = (1.0 + 5.0 ** 0.5) / 2.0


def e8_roots() -> np.ndarray:
    """The 240 roots of E8 in the standard coordinates, shape (240, 8)."""
    roots = []
    for i, j in itertools.combinations(range(8), 2):
        for si, sj in itertools.product((1.0, -1.0), repeat=2):
            v = np.zeros(8)
            v[i], v[j] = si, sj
            roots.append(v)
    for signs in itertools.product((0.5, -0.5), repeat=8):
        if sum(1 for s in signs if s < 0) % 2 == 0:
            roots.append(np.array(signs))
    return np.array(roots)


def simple_roots() -> np.ndarray:
    """Bourbaki simple roots of E8 in the coordinates of e8_roots(), shape (8, 8)."""
    a = np.zeros((8, 8))
    a[0] = 0.5 * np.array([1, -1, -1, -1, -1, -1, -1, 1])
    a[1] = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    a[2] = np.array([-1, 1, 0, 0, 0, 0, 0, 0])
    a[3] = np.array([0, -1, 1, 0, 0, 0, 0, 0])
    a[4] = np.array([0, 0, -1, 1, 0, 0, 0, 0])
    a[5] = np.array([0, 0, 0, -1, 1, 0, 0, 0])
    a[6] = np.array([0, 0, 0, 0, -1, 1, 0, 0])
    a[7] = np.array([0, 0, 0, 0, 0, -1, 1, 0])
    return a


def reflection(alpha: np.ndarray) -> np.ndarray:
    """Matrix of the reflection in the hyperplane orthogonal to alpha."""
    return np.eye(len(alpha)) - 2.0 * np.outer(alpha, alpha) / float(alpha @ alpha)


def coxeter_element() -> np.ndarray:
    """Product of the simple reflections, in order, shape (8, 8)."""
    c = np.eye(8)
    for alpha in simple_roots():
        c = reflection(alpha) @ c
    return c


def h4_projection() -> Tuple[np.ndarray, np.ndarray]:
    """Return (P_phys, P_perp), each 4x8 with orthonormal rows.

    The rows span the invariant planes of the Coxeter element for eigenvalue angles {12, 132} deg (physical) and
    {84, 156} deg (internal)."""
    w, v = np.linalg.eig(coxeter_element())
    ang = np.degrees(np.angle(w))
    planes = {}
    for k in range(8):
        if ang[k] > 1e-6:                               # one eigenvector per conjugate pair
            vec = v[:, k]
            re, im = np.sqrt(2.0) * vec.real, np.sqrt(2.0) * vec.imag
            planes[int(round(ang[k]))] = (re, im)
    phys = [planes[12], planes[132]]
    perp = [planes[84], planes[156]]
    p_phys = np.array([x for pair in phys for x in pair])
    p_perp = np.array([x for pair in perp for x in pair])
    return p_phys, p_perp


def model_set(n: int = 2, radius: float = 1.0) -> np.ndarray:
    """Physical (4D) images of the E8 lattice points sum n_i alpha_i, |n_i| <= n, whose internal image has
    norm <= radius.  Spherical window: quasiperiodic, but not the exact Elser-Sloane window."""
    alpha = simple_roots()
    rng = np.arange(-n, n + 1)
    coeffs = np.array(np.meshgrid(*([rng] * 8), indexing="ij")).reshape(8, -1).T
    pts = coeffs @ alpha
    p_phys, p_perp = h4_projection()
    keep = np.linalg.norm(pts @ p_perp.T, axis=1) <= radius + 1e-12
    return pts[keep] @ p_phys.T


def summary() -> str:
    roots = e8_roots()
    p_phys, p_perp = h4_projection()
    r_phys = np.linalg.norm(roots @ p_phys.T, axis=1)
    radii = np.unique(np.round(r_phys, 9))
    lines = [f"E8 roots: {len(roots)}",
             f"projected radii (physical space): {radii.tolist()}",
             f"ratio of the two radii: {radii[-1] / radii[0]:.9f}   (golden ratio {PHI:.9f})"]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
