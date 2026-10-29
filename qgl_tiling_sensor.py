#!/usr/bin/env python3
"""
qgl_tiling_sensor.py -- a Jaynes-Cummings sensor driven along a line through a Chair44 tiling.

Question.  The hull theorem for Chair44 (Romanov; Spectre repository) says that the tilings form a minimal, uniquely
ergodic family, so Birkhoff averages over large *boxes* converge uniformly to the same value in every tiling.  Does a
sensor that moves through the tiling along a *line* therefore dephase more slowly than a sensor moving through a
random decoration with the same one-cell statistics?

Model.  The sensor visits cells c_0, c_1, ... along a line and spends the time dt in each.  In cell i the
Jaynes-Cummings coupling is

        g_i = g0 (1 + eps m_i),         m_i = (observable of cell i, centred and scaled to unit variance),

with the observable either 'pyramid' (the summed signed pyramid heights of the cell, 0 on five of the seven native cells and +-36
on the other two) or 'notch_x/y/z' (a component of the notch direction of the chair that owns the cell).  The atom
starts in (|0,g> + |0,e>)/sqrt 2.  In the dispersive regime the excited branch picks up the phase chi_i dt with
chi_i = g_i^2 / Delta, so the phase of the atomic coherence relative to an unmodulated sensor is a Birkhoff sum
        phi_L = (2 chi0 dt eps) sum_{i<L} m_i + O(eps^2).
One run along a given line returns a pure phase.  The sensor does not know where on the line it started, so the
state it reports is the average over starts; its coherence is
        R(L) = | mean_s <e|rho_s(L)|g> | / | <e|rho_unmodulated(L)|g> |,
and for Gaussian phases R = exp(-Var phi / 2).  Everything is run through `LindbladJCEngine` (qgl_jc_lindblad), so
photon loss and atomic decay are the engine's own and cancel in the ratio.

Controls.  'shuffled' is the same cube with its cells permuted at random: the same one-cell statistics, no spatial
structure.  The tiling cubes come from qgl_tiling.label_cube.

Two sensor designs.  modulate='g' (coupling follows the observable, as first proposed) and modulate='omega_a' (atomic
frequency follows it).  The dispersive phase is quadratic in g, so the coupling design also responds to m^2, and
sum(m^2 - 1) is NOT bounded along an axis (each axis line meets its own fraction of the two nonzero cell types): in
that design coherence on the axes stops decaying but then falls again (0.76 at L = 128).  The frequency design is
linear and shows the effect cleanly.

Result (see main(), figures/tiling_sensor.png, tests/test_tiling.py).  The prediction in the form "every line
dephases more slowly" is false: the hull theorem controls averages over boxes (the whole Z^3 action), and a line is a
one-dimensional sub-action whose orbit closure in the internal space Z_2^3 is only a one-dimensional subgroup.  What
a line sees depends on its direction:
  * pyramid observable, axes and face diagonals: line sums bounded (sup 108 = 3 x 36, 144), Var/L -> 0.001, coherence
    0.99 at L = 128 against 0.1 for the shuffled control;
  * pyramid along the body diagonal and notch_z along x / face diagonal: ballistic (Var ~ L^2), coherence collapses
    faster than random (later 'revivals' are phase wrapping of a few discrete line means, not coherence);
  * notch_z along z: bounded (existing paper test), coherence 1.00;
  * rational oblique and generic (irrational) directions: Var/L ~ 1 (notch_z generic 0.5 at L = 128), coherence equal
    to the control within noise except a mild gain for notch_z generic.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

import qgl_tiling as T
from unified_sandbox_core import Config
from qgl_jc_lindblad import LindbladJCEngine

TWO_PI = 2.0 * np.pi

DIRECTIONS: Dict[str, Tuple[float, float, float]] = {
    "x axis (1,0,0)": (1.0, 0.0, 0.0),
    "z axis (0,0,1)": (0.0, 0.0, 1.0),
    "face diagonal (1,1,0)": (1.0, 1.0, 0.0),
    "body diagonal (1,1,1)": (1.0, 1.0, 1.0),
    "rational (1,1/3,2/3)": (1.0, 1.0 / 3.0, 2.0 / 3.0),
    "generic (1,sqrt2-1,sqrt3-1)": (1.0, 2.0 ** 0.5 - 1.0, 3.0 ** 0.5 - 1.0),
}


# --------------------------------------------------------------------------
#  Lines through a cube of cells
# --------------------------------------------------------------------------

def line_indices(direction: Sequence[float], n_lines: int, length: int, side: int,
                 rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell indices of n_lines lines.  Step i of a line with direction (a,b,c) (components in [0,1]) is the cell
    start + floor(u + i (a,b,c)) with u uniform in [0,1)^3, so lattice directions give exact lattice lines and
    irrational ones give the usual digitised line.  Starts are uniform in the lower half of the cube so every line
    stays inside the cube."""
    n = np.arange(length)[None, :]
    half = side // 2
    idx = []
    for comp in direction:
        start = rng.integers(0, half, n_lines)[:, None]
        u = rng.random((n_lines, 1))
        idx.append(start + np.floor(u + n * comp).astype(int))
    assert all(i.max() < side for i in idx), "line leaves the cube"
    return tuple(idx)


def centred_observable(cube_labels: np.ndarray, name: str, rng: np.random.Generator,
                       shuffled: bool) -> Tuple[np.ndarray, float]:
    """Observable field of the cube, centred on its mean (the hull-theorem mean for the pyramid is 0 exactly), and
    its standard deviation.  `shuffled` permutes the cells."""
    F = T.field(cube_labels, name).astype(np.float64)
    F -= F.mean()
    if shuffled:
        flat = F.ravel().copy()
        rng.shuffle(flat)
        F = flat.reshape(F.shape)
    return F, float(F.std())


# --------------------------------------------------------------------------
#  Line-sum statistics
# --------------------------------------------------------------------------

def sum_statistics(F: np.ndarray, direction, lengths: Sequence[int], n_lines: int,
                   rng: np.random.Generator) -> Dict[int, Dict[str, float]]:
    """Variance (in units of L times the one-cell variance) and sup-norm of the prefix sums of F along random lines."""
    L = max(lengths)
    X, Y, Z = line_indices(direction, n_lines, L, F.shape[0], rng)
    S = np.cumsum(F[X, Y, Z], axis=1)
    v0 = F.var()
    return {l: dict(var_over_L=float(S[:, l - 1].var() / (l * v0)),
                    sup=float(np.abs(S[:, :l]).max()),
                    mean_line_mean=float(np.abs(S[:, l - 1].mean()) / l)) for l in lengths}


# --------------------------------------------------------------------------
#  Sensor
# --------------------------------------------------------------------------

def sensor_config(n_cavity: int = 3) -> Config:
    """Dispersive regime: Delta = 0.3 * 2 pi, g0 = 0.09 * 2 pi (the repository defaults), small decay."""
    cfg = Config()
    cfg.n_cavity = n_cavity
    cfg.omega_atom_0 = TWO_PI * 1.3
    cfg.omega_cavity_0 = TWO_PI * 1.0
    cfg.kappa_0 = 0.002
    cfg.gamma_0 = 0.001
    return cfg


def dispersive_shift(cfg: Config) -> float:
    return cfg.g_0 ** 2 / (cfg.omega_atom_0 - cfg.omega_cavity_0)


def eps_for_target(cfg: Config, dt: float, phase_std_per_cell: float, modulate: str = "g") -> float:
    """eps such that, for unit-variance m, the linearised phase increment per cell has the given standard deviation.
    modulate='g':       g_i = g0 (1 + eps m_i);        phase ~ -(dt/Delta) g_i^2, linear part 2 chi0 dt eps m_i
    modulate='omega_a': w_i = w_a (1 + eps m_i);       relative phase of e and g ~ dt w_i (+ a second-order effect through Delta)"""
    if modulate == "g":
        return phase_std_per_cell / (2.0 * dispersive_shift(cfg) * dt)
    if modulate == "omega_a":
        return phase_std_per_cell / (cfg.omega_atom_0 * dt)
    raise ValueError(modulate)


def run_sensor(m: np.ndarray, cfg: Config, eps: float, dt: float, checkpoints: Sequence[int],
               modulate: str = "g") -> np.ndarray:
    """Drive one LindbladJCEngine through the modulation sequence m (shape (L,)).  Returns the complex atomic
    coherence sum_n <n,e|rho|n,g> at the checkpoints."""
    eng = LindbladJCEngine(cfg)
    eng.rho[:] = 0.0
    eng.rho[0, 0] = eng.rho[1, 1] = eng.rho[0, 1] = eng.rho[1, 0] = 0.5
    out = []
    cps = set(checkpoints)
    p = dict(omega_c=cfg.omega_cavity_0, omega_a=cfg.omega_atom_0, kappa=cfg.kappa_0, gamma=cfg.gamma_0)
    for i, mi in enumerate(m, start=1):
        p["g"] = cfg.g_0
        p["omega_a"] = cfg.omega_atom_0
        p["g" if modulate == "g" else "omega_a"] *= 1.0 + eps * float(mi)
        eng.evolve(p, dt)
        if i in cps:
            out.append(sum(eng.rho[2 * n + 1, 2 * n] for n in range(eng.n_cavity)))
    return np.array(out)


def coherence_curve(F_unit: np.ndarray, direction, lengths: Sequence[int], n_lines: int, cfg: Config,
                    eps: float, dt: float, rng: np.random.Generator, modulate: str = "g") -> Dict[str, np.ndarray]:
    """Ensemble coherence R(L) of the sensor over n_lines random starts on lines in `direction` through the unit-variance
    field F_unit, and the circular phase-only version."""
    L = max(lengths)
    X, Y, Z = line_indices(direction, n_lines, L, F_unit.shape[0], rng)
    ref = run_sensor(np.zeros(L), cfg, 0.0, dt, lengths, modulate)
    C = np.array([run_sensor(F_unit[X[k], Y[k], Z[k]], cfg, eps, dt, lengths, modulate) for k in range(n_lines)])
    R = np.abs(C.mean(axis=0)) / np.abs(ref)
    Rphase = np.abs(np.exp(1j * (np.angle(C) - np.angle(ref)[None, :])).mean(axis=0))
    err = np.abs(C / ref[None, :]).std(axis=0) / np.sqrt(n_lines)
    return dict(R=R, R_phase=Rphase, err=err)


# --------------------------------------------------------------------------
#  Experiment
# --------------------------------------------------------------------------

def experiment(observable: str = "pyramid", depth: int = 8, seed_label: int = 5,
               lengths: Sequence[int] = (4, 8, 16, 32, 64, 128), n_lines: int = 1000,
               phase_std_per_cell: float = 0.22, dt: float = 6.5, directions=None, seed: int = 11,
               modulate: str = "g", verbose: bool = True) -> Dict:
    rng = np.random.default_rng(seed)
    cfg = sensor_config()
    eps = eps_for_target(cfg, dt, phase_std_per_cell, modulate)
    cube = T.label_cube(seed_label, depth)
    Ft, sd = centred_observable(cube, observable, rng, shuffled=False)
    Fs, _ = centred_observable(cube, observable, rng, shuffled=True)
    Ft_u, Fs_u = Ft / sd, Fs / sd
    res = {"observable": observable, "modulate": modulate, "depth": depth, "seed_label": seed_label, "eps": eps, "dt": dt,
           "chi0": dispersive_shift(cfg), "lengths": list(lengths), "n_lines": n_lines, "directions": {}}
    for name, d in (directions or DIRECTIONS).items():
        t0 = time.time()
        st_t = sum_statistics(Ft, d, lengths, 4000, rng)
        st_s = sum_statistics(Fs, d, lengths, 4000, rng)
        ct = coherence_curve(Ft_u, d, lengths, n_lines, cfg, eps, dt, rng, modulate)
        cs = coherence_curve(Fs_u, d, lengths, n_lines, cfg, eps, dt, rng, modulate)
        res["directions"][name] = dict(
            tiling=dict(R=ct["R"].tolist(), R_phase=ct["R_phase"].tolist(), err=ct["err"].tolist(),
                        var_over_L=[st_t[l]["var_over_L"] for l in lengths], sup=[st_t[l]["sup"] for l in lengths]),
            shuffled=dict(R=cs["R"].tolist(), R_phase=cs["R_phase"].tolist(), err=cs["err"].tolist(),
                          var_over_L=[st_s[l]["var_over_L"] for l in lengths], sup=[st_s[l]["sup"] for l in lengths]))
        if verbose:
            print(f"[{observable}/{modulate}] {name:28s} ({time.time() - t0:4.0f}s)")
            print("   L          " + " ".join(f"{l:7d}" for l in lengths))
            for lab, key in (("tiling", "tiling"), ("shuffled", "shuffled")):
                r = res["directions"][name][key]
                print(f"   R {lab:9s}" + " ".join(f"{x:7.3f}" for x in r["R"]))
            for lab, key in (("tiling", "tiling"), ("shuffled", "shuffled")):
                r = res["directions"][name][key]
                print(f"   Var/(L v) {lab:8s}" + " ".join(f"{x:7.3f}" for x in r["var_over_L"]))
    return res


def main():
    out = {}
    for mod in ("g", "omega_a"):
        for obs in ("pyramid", "notch_z"):
            out[f"{obs}/{mod}"] = experiment(obs, modulate=mod)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "tiling_sensor.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
