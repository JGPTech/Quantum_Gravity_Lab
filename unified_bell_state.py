#!/usr/bin/env python3
"""
Quantum Gravity Lab — BELL STATE EDITION
=========================================

An entanglement-focused interface exploring quantum correlations in curved spacetime:

  • Stage 1: EntanglementConsole - Configure orbital + Bell measurement settings
  • Stage 2: run_entangled_simulation - Extended pipeline tracking Bell correlations
  • Stage 3: BellCorrelationViewer - Quantum information dashboard

Key additions:
  - Bell parameter evolution β(t) tied to spacetime curvature
  - Entanglement entropy S(t) between qubit and cavity
  - CHSH inequality monitoring with violation regions
  - Quantum discord and concurrence tracking
  - Phase-space + Bloch-sphere visualization

The core insight: gravitational fields modify quantum correlations,
creating position-dependent Bell violation patterns that encode
the underlying spacetime geometry.
"""

import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3D
from scipy.linalg import expm

# Core imports from your unified sandbox
from unified_sandbox_core import (
    Config,
    canonical_periapsis_ic,
    run_unified_trajectory,
    compute_quantum_clock_rate,
)

# Calibration helper (your existing module)
from calibrate import calibrate_complexity_power

# ======================================================================
#  THEME — QUANTUM INFORMATION PALETTE
# ======================================================================
THEME = {
    "bg": "#0a0f1c",          # Deep quantum void
    "panel": "#162238",       # Panel background
    "fg": "#e8f4f8",          # Main text
    "bell_blue": "#00d4ff",   # Bell state primary
    "bell_purple": "#b366ff", # Entanglement
    "bell_gold": "#ffd700",   # Classical limit
    "violation": "#ff0066",   # Bell violation zones
    "classical": "#66ff66",   # Classical correlations
    "grid": "#2a3f5f",
    "font": "monospace",
}

plt.rcParams.update({
    "font.family": THEME["font"],
    "text.color": THEME["fg"],
    "axes.facecolor": THEME["bg"],
    "axes.edgecolor": THEME["bell_blue"],
    "axes.labelcolor": THEME["fg"],
    "axes.titlecolor": THEME["bell_blue"],
    "xtick.color": THEME["bell_blue"],
    "ytick.color": THEME["bell_blue"],
    "grid.color": THEME["grid"],
    "figure.facecolor": THEME["bg"],
})

# ======================================================================
#  CALIBRATION HELPERS
# ======================================================================

CALIBRATION_CSV = "complexity_power_grid_scan.csv"


def load_calibration_df() -> pd.DataFrame:
    """Load calibration CSV if it exists, else return empty DataFrame."""
    if os.path.exists(CALIBRATION_CSV):
        return pd.read_csv(CALIBRATION_CSV)

    cols = [
        "p", "e", "n_orbits", "n_output", "ok", "complexity_power",
        "rel_error", "measured_shift", "theory_shift", "n_peri",
        "k_rq_ratio", "r_q_transition",
    ]
    return pd.DataFrame(columns=cols)


def save_calibration_df(df: pd.DataFrame) -> None:
    df.to_csv(CALIBRATION_CSV, index=False)


def find_calibrated_row(df: pd.DataFrame, p: float, e: float) -> Optional[pd.Series]:
    """Return best existing calibration row for (p,e), or None."""
    if df.empty:
        return None
    rows = df[(np.isclose(df["p"], p)) & (np.isclose(df["e"], e))]
    if rows.empty:
        return None
    idx = rows["rel_error"].abs().idxmin()
    return rows.loc[idx]


def get_or_calibrate_complexity_power(
    p: float,
    e: float,
    df_calib: pd.DataFrame,
    n_orbits_calib: int = 8,
    n_output_calib: int = 8000,
) -> Tuple[float, pd.DataFrame, dict]:
    """
    Look up complexity_power for (p,e) in df_calib.
    If not present, call calibrate_complexity_power and append the row.
    """
    row = find_calibrated_row(df_calib, p, e)
    if row is not None and bool(row.get("ok", True)):
        cp = float(row["complexity_power"])
        record = row.to_dict()
        return cp, df_calib, record

    print(f"\n[CAL] No record for (p={p:.3f}, e={e:.3f}) → running calibrate_complexity_power")

    res = calibrate_complexity_power(
        p=p,
        e=e,
        n_orbits=n_orbits_calib,
        n_output=n_output_calib,
        coarse_range=(0.2, 2.5),
        coarse_steps=9,
        fine_half_width=0.20,
        fine_steps=40,
    )

    record = {
        "p": float(res.get("p", p)),
        "e": float(res.get("e", e)),
        "n_orbits": int(n_orbits_calib),
        "n_output": int(n_output_calib),
        "ok": bool(res.get("ok", False)),
        "complexity_power": float(res.get("complexity_power", np.nan)),
        "rel_error": float(res.get("rel_error", np.nan)),
        "measured_shift": float(res.get("measured_shift", np.nan)),
        "theory_shift": float(res.get("theory_shift", np.nan)),
        "n_peri": int(res.get("n_peri", 0)),
        "k_rq_ratio": float(res.get("k_rq_ratio", np.nan)),
        "r_q_transition": float(res.get("r_q_transition", np.nan)),
    }

    df_calib = pd.concat([df_calib, pd.DataFrame([record])], ignore_index=True)
    save_calibration_df(df_calib)

    cp = float(record["complexity_power"])
    print(f"[CAL] Stored calibration: complexity_power = {cp:.6f}")
    return cp, df_calib, record

# ======================================================================
#  QUANTUM CORRELATION FUNCTIONS (BELL / CHSH)
# ======================================================================


def compute_bell_parameter(C_Q: float, r: float, v2: float, cfg: Config) -> float:
    """
    Bell parameter β ∈ [2, 2√2] encoding correlation strength.

    At C_Q = 1 (quantum): β → 2√2 (maximal violation)
    At C_Q = 0 (classical): β → 2 (no violation)
    """
    # relativistic gamma from speed
    gamma = 1.0 / np.sqrt(max(1.0 - v2 / (cfg.c ** 2), 1e-6))

    # gravitational suppression (static Schwarzschild redshift)
    r_g = cfg.r_g
    grav_factor = math.sqrt(max(1.0 - 2.0 * r_g / max(r, 1e-12), 0.0))

    beta_classical = 2.0
    beta_quantum = 2.0 * math.sqrt(2.0)

    eff_C_Q = C_Q * grav_factor * (2.0 - 1.0 / gamma)
    eff_C_Q = float(np.clip(eff_C_Q, 0.0, 1.0))

    return beta_classical + eff_C_Q * (beta_quantum - beta_classical)


def compute_chsh_bounds(beta: float) -> Tuple[float, float, bool]:
    """
    CHSH inequality: |S| ≤ 2 classically, can reach 2√2 quantumly.

    Returns: (classical_bound, quantum_bound, is_violated)
    """
    classical_bound = 2.0
    quantum_bound = 2.0 * math.sqrt(2.0)
    is_violated = beta > classical_bound
    return classical_bound, quantum_bound, is_violated

# ======================================================================
#  JC OPERATORS + ENTANGLEMENT HELPERS
# ======================================================================

def build_jc_operators(n_cavity: int):
    """
    Build Jaynes–Cummings operators in basis
      |g,0>, |e,0>, |g,1>, |e,1>, ...
    dim = 2 * n_cavity.
    """
    # cavity ladder + number
    a = np.zeros((n_cavity, n_cavity), dtype=np.complex128)
    for n in range(1, n_cavity):
        a[n - 1, n] = math.sqrt(n)
    a_dag = a.conj().T
    n_op = a_dag @ a

    # qubit Pauli operators
    sigma_x = np.array([[0, 1],
                        [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j],
                        [1j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0],
                        [0, -1]], dtype=np.complex128)
    sigma_plus = np.array([[0, 1],
                           [0, 0]], dtype=np.complex128)
    sigma_minus = np.array([[0, 0],
                            [1, 0]], dtype=np.complex128)

    I_c = np.eye(n_cavity, dtype=np.complex128)
    I_q = np.eye(2, dtype=np.complex128)

    # full-space operators
    N_full = np.kron(n_op, I_q)
    Sz_full = np.kron(I_c, sigma_z)
    Sx_full = np.kron(I_c, sigma_x)
    Sy_full = np.kron(I_c, sigma_y)

    a_sigma_plus = np.kron(a, sigma_plus)
    adag_sigma_min = np.kron(a_dag, sigma_minus)

    return {
        "N": N_full,
        "Sz": Sz_full,
        "Sx": Sx_full,
        "Sy": Sy_full,
        "a_sigma_plus": a_sigma_plus,
        "adag_sigma_min": adag_sigma_min,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "sigma_z": sigma_z,
    }

def init_entangled_jc_state(n_cavity: int) -> np.ndarray:
    """
    Bell-like product state giving a Bloch vector on the equator:
        |ψ> = (|g,0> + |e,0>) / √2
    in basis |g,0>, |e,0>, |g,1>, |e,1>, ...
    """
    dim = 2 * n_cavity
    psi = np.zeros(dim, dtype=np.complex128)

    # |g,0> and |e,0> with equal weight
    psi[0] = 1.0 / math.sqrt(2.0)  # |g,0>
    psi[1] = 1.0 / math.sqrt(2.0)  # |e,0>

    rho = np.outer(psi, psi.conj())
    return rho

def reduced_qubit_density(rho: np.ndarray, n_cavity: int) -> np.ndarray:
    """
    Trace out cavity → 2x2 reduced qubit state.

    index = 2*n + q, n ∈ [0..n_cavity-1], q=0(g),1(e)
    """
    rho_q = np.zeros((2, 2), dtype=np.complex128)

    for n in range(n_cavity):
        ig = 2 * n
        ie = ig + 1
        if ie >= rho.shape[0]:
            break
        rho_q[0, 0] += rho[ig, ig]
        rho_q[1, 1] += rho[ie, ie]
        rho_q[0, 1] += rho[ig, ie]
        rho_q[1, 0] += rho[ie, ig]

    tr = np.trace(rho_q)
    if abs(tr) > 1e-12:
        rho_q /= tr
    return rho_q


def entanglement_entropy_qubit(rho_q: np.ndarray) -> float:
    """Von Neumann entropy S = -Tr(ρ log₂ ρ) for 2x2 state."""
    eigvals = np.linalg.eigvalsh(rho_q)
    eigvals = eigvals[eigvals > 1e-12]
    if eigvals.size == 0:
        return 0.0
    return float(-np.sum(eigvals * np.log2(eigvals)))


def concurrence_from_rhoq(rho_q: np.ndarray) -> float:
    """
    Coherence-like concurrence proxy: C ≈ 2|ρ_01| ∈ [0,1].
    """
    c = 2.0 * abs(rho_q[0, 1])
    return float(np.clip(c, 0.0, 1.0))


def discord_from_rhoq(rho_q: np.ndarray, C_Q: float) -> float:
    """
    Toy "discord" proxy: entanglement entropy modulated by complexity.
    """
    S = entanglement_entropy_qubit(rho_q)
    return float(S * (1.0 - math.exp(-3.0 * C_Q)))

# ======================================================================
#  EXTENDED SIMULATION RESULTS
# ======================================================================


@dataclass
class RunConfig:
    """Extended with Bell measurement settings."""
    p: float
    e: float
    n_orbits: int
    n_output: int
    s_rho: float = 1.0
    s_phi: float = 0.6
    s_v: float = 0.4
    with_jc: bool = True
    use_quantum_time: bool = True
    # Bell-specific (not yet deeply used, but kept for UI completeness)
    track_bell: bool = True
    measurement_basis: str = "XY"  # XY, XZ, or YZ
    discord_samples: int = 100


@dataclass
class BellSimulationResults:
    """Extended results with quantum correlation tracking."""
    # Base trajectory data
    run_config: RunConfig
    complexity_power: float
    k_rq_ratio: Optional[float]
    r_q_transition: float
    alpha_gr: float

    # Kinematics
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    r: np.ndarray
    v2: np.ndarray

    # Quantum complexity
    C_Q: np.ndarray
    clock_q: np.ndarray
    clock_gr: np.ndarray

    # Energy/momentum
    E: np.ndarray
    L: np.ndarray

    # GR metrics
    peri_angles: np.ndarray
    precession_measured: float
    precession_theory: float

    # JC system
    bloch: np.ndarray        # (N, 3) Bloch vectors
    photons: np.ndarray      # ⟨n⟩(t)
    coherence: np.ndarray    # |ρ_01|(t)

    # Bell additions
    bell_parameter: np.ndarray          # β(t) ∈ [2, 2√2]
    entanglement_entropy: np.ndarray    # S(t)
    concurrence: np.ndarray             # C(t)
    quantum_discord: np.ndarray         # D(t)
    bell_violation_regions: List[Tuple[float, float]]  # Time intervals
    max_violation: float
    total_violation_fraction: float

# ======================================================================
#  SIMULATION RUNNER WITH BELL TRACKING
# ======================================================================


def run_entangled_simulation(run_cfg: RunConfig, status_cb=None) -> BellSimulationResults:
    """
    Extended pipeline tracking Bell correlations throughout orbit.
    """

    def log(msg: str):
        if status_cb:
            status_cb(msg)
        print(f"[BELL] {msg}")

    # ------------------------------------------------------------------
    # Configure + calibrate
    # ------------------------------------------------------------------
    log("Initializing quantum correlation trackers...")

    df_calib = load_calibration_df()
    cp, df_calib, _ = get_or_calibrate_complexity_power(
        run_cfg.p, run_cfg.e, df_calib, 8, 8000
    )

    cfg = Config()
    cfg.p = float(run_cfg.p)
    cfg.e = float(run_cfg.e)
    cfg.complexity_power = float(cp)

    # Dynamic k_rq_ratio if enabled in Config
    k_rq_ratio = None
    if getattr(cfg, "k_rq_ratio_mode", "fixed") == "dynamic":
        from unified_sandbox_core import k_rq_from_pe
        cfg.k_rq_ratio = k_rq_from_pe(cfg.p, cfg.e, default=cfg.k_rq_ratio_default)
        k_rq_ratio = float(cfg.k_rq_ratio)

    # Setup orbit
    L_sq = cfg.G * cfg.M * cfg.p
    cfg.alpha_gr = (3.0 * cfg.G * cfg.M * L_sq) / (cfg.c ** 2)
    if hasattr(cfg, "update_r_q_transition_from_orbit"):
        cfg.update_r_q_transition_from_orbit()

    alpha_gr = float(cfg.alpha_gr)
    r_q_transition = float(getattr(cfg, "r_q_transition", np.nan))

    # Time span
    a = cfg.p / (1.0 - cfg.e ** 2)
    T_orb = 2.0 * math.pi * math.sqrt(a ** 3 / (cfg.G * cfg.M))
    t_max = run_cfg.n_orbits * T_orb

    # ------------------------------------------------------------------
    # Run trajectory
    # ------------------------------------------------------------------
    log("Integrating entangled trajectory...")
    r0, v0 = canonical_periapsis_ic(cfg)

    df = run_unified_trajectory(
        cfg, r0, v0, t_max,
        n_output=run_cfg.n_output,
        with_jc=run_cfg.with_jc,
        use_quantum_time=run_cfg.use_quantum_time,
    )

    # Extract base data
    t = df["t"].values
    x = df["x"].values
    y = df["y"].values
    z = df["z"].values
    vx = df["vx"].values
    vy = df["vy"].values
    vz = df["vz"].values

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    v2 = vx ** 2 + vy ** 2 + vz ** 2

    # ------------------------------------------------------------------
    # Complexity & clocks & Bell β(t)
    # ------------------------------------------------------------------
    log("Computing quantum correlations along trajectory...")

    C_Q_list = []
    clock_q_list = []
    clock_gr_list = []
    bell_params = []

    for i in range(len(t)):
        ri = r[i]
        v2i = v2[i]

        # your existing compute_quantum_clock_rate signature:
        # returns (something, C_Q, clock_q)
        _, C_Q_i, clock_i = compute_quantum_clock_rate(ri, cfg)

        # --- SAFETY: kill NaNs from the clock model ---
        if not np.isfinite(C_Q_i):
            C_Q_i = 0.0
        if not np.isfinite(clock_i):
            clock_i = 1.0

        C_Q_list.append(float(C_Q_i))
        clock_q_list.append(float(clock_i))

        # Schwarzschild clock
        r_g = cfg.r_g
        x_sch = 1.0 - 2.0 * r_g / max(ri, 1e-12)
        clock_gr = math.sqrt(max(x_sch, 0.0))
        clock_gr_list.append(clock_gr)

        beta = compute_bell_parameter(C_Q_i, ri, v2i, cfg)
        bell_params.append(beta)

    C_Q = np.array(C_Q_list)
    clock_q = np.array(clock_q_list)
    clock_gr = np.array(clock_gr_list)
    bell_parameter = np.array(bell_params)

    # ------------------------------------------------------------------
    # Energy & angular momentum
    # ------------------------------------------------------------------
    E = 0.5 * v2 - cfg.G * cfg.M / np.maximum(r, 1e-8)
    L_arr = np.zeros_like(r)
    for i in range(len(r)):
        L_vec = np.cross([x[i], y[i], z[i]], [vx[i], vy[i], vz[i]])
        L_arr[i] = np.linalg.norm(L_vec)

    # ------------------------------------------------------------------
    # JC evolution with Bell tracking (real dynamics)
    # ------------------------------------------------------------------
    log("Evolving entangled cavity states...")

    n_cavity = 15      # keep dim modest so expm isn't insane

    ops = build_jc_operators(n_cavity)
    N_full = ops["N"]
    Sz_full = ops["Sz"]
    Sx_full = ops["Sx"]
    Sy_full = ops["Sy"]
    a_sigma_plus = ops["a_sigma_plus"]
    adag_sigma_min = ops["adag_sigma_min"]

    # base frequencies (tunable)
    omega_c0 = 1.0
    omega_a0 = 1.0
    g0 = 0.05

    rho = init_entangled_jc_state(n_cavity)

    # precompute orbital phase for the drive (tie qubit precession to geometry)
    orbital_phase = np.unwrap(np.arctan2(y, x))

    bloch_vecs: List[List[float]] = []
    photons: List[float] = []
    coherences: List[float] = []
    entropies: List[float] = []
    concurrences: List[float] = []
    discords: List[float] = []

    dt_array = np.diff(t) if len(t) > 1 else np.array([0.0])
    dt_mean = float(np.mean(dt_array)) if dt_array.size > 0 else 0.0

    for i in range(len(t)):
        if i > 0 and dt_mean > 0.0:
            CQi = C_Q[i]
            ri = r[i]

            # gravitational redshift factor
            r_g = cfg.r_g
            grav = math.sqrt(max(1.0 - 2.0 * r_g / max(ri, 1e-12), 0.0))
            omega_c = omega_c0 * grav
            omega_a = omega_a0 * grav
            g_eff = g0 * (0.3 + 0.7 * CQi)

            # --- SAFETY: clamp crazy / NaN couplings ---
            if not np.isfinite(g_eff):
                g_eff = 0.0

            # --- JC Hamiltonian ---
            H0 = (
                omega_c * N_full
                + 0.5 * omega_a * Sz_full
                + g_eff * (a_sigma_plus + adag_sigma_min)
            )

            # --- Added: qubit drive to get full Bloch precession ---
            # Drive strength modulated by complexity, phase locked to the orbital angle.
            drive_amp = 0.40 * (0.3 + 0.7 * CQi)      # tweak 0.20 up/down to taste
            phi_drive = float(orbital_phase[i])

            H_drive = drive_amp * (
                math.cos(phi_drive) * Sx_full +
                math.sin(phi_drive) * Sy_full
            )

            H = H0 + H_drive

            dt_local = float(t[i] - t[i - 1]) if i < len(t) else dt_mean

            U = expm(-1j * H * dt_local)
            rho = U @ rho @ U.conj().T

            # --- numerical hygiene + NaN scrub ---
            rho = 0.5 * (rho + rho.conj().T)  # enforce Hermiticity

            if not np.all(np.isfinite(rho)):
                rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)

            tr = np.trace(rho)
            if abs(tr) > 1e-12:
                rho /= tr
            else:
                rho = init_entangled_jc_state(n_cavity)

        # diagnostics at time i
        rho_q = reduced_qubit_density(rho, n_cavity)

        S = entanglement_entropy_qubit(rho_q)
        Cc = concurrence_from_rhoq(rho_q)
        D = discord_from_rhoq(rho_q, C_Q[i])

        entropies.append(S)
        concurrences.append(Cc)
        discords.append(D)

        # Bloch vector from rho_q (no unit normalization: radius = purity)
        sx = 2.0 * np.real(rho_q[0, 1])
        sy = -2.0 * np.imag(rho_q[0, 1])
        sz = np.real(rho_q[0, 0] - rho_q[1, 1])

        norm = math.sqrt(sx * sx + sy * sy + sz * sz)

        # numerical hygiene
        if not np.isfinite(norm) or norm < 1e-12:
            sx, sy, sz = 0.0, 0.0, 0.0
        elif norm > 1.0 + 1e-6:
            # clamp pathological >1 cases back onto the sphere
            sx /= norm
            sy /= norm
            sz /= norm

        bloch_vecs.append([sx, sy, sz])


        # photon number and simple coherence norm
        n_mean = np.real(np.trace(rho @ N_full))
        photons.append(float(n_mean))

        coh = float(abs(rho_q[0, 1]))
        coherences.append(coh)

    entanglement_entropy = np.array(entropies)
    concurrence = np.array(concurrences)
    quantum_discord = np.array(discords)
    bloch = np.array(bloch_vecs)
    if bloch.size > 0:
        mask_bad = ~np.isfinite(bloch).all(axis=1)
        if np.any(mask_bad):
            print(f"[JC] {mask_bad.sum()} non-finite Bloch samples → clamping to (0,0,1)")
            bloch[mask_bad] = np.array([0.0, 0.0, 1.0])

    photons = np.array(photons)
    coherence = np.array(coherences)

    # ------------------------------------------------------------------
    # Bell violation regions in time
    # ------------------------------------------------------------------
    log("Identifying Bell violation regions...")
    violation_regions: List[Tuple[float, float]] = []
    in_violation = False
    start_t = 0.0

    for i, beta in enumerate(bell_parameter):
        _, _, is_violated = compute_chsh_bounds(beta)

        if is_violated and not in_violation:
            start_t = t[i]
            in_violation = True
        elif not is_violated and in_violation:
            violation_regions.append((start_t, t[i]))
            in_violation = False

    if in_violation:
        violation_regions.append((start_t, t[-1]))

    max_violation = float(np.max(bell_parameter) - 2.0)
    violation_mask = bell_parameter > 2.0
    total_violation_fraction = (
        float(np.sum(violation_mask)) / len(bell_parameter) if len(bell_parameter) > 0 else 0.0
    )

    # ------------------------------------------------------------------
    # Precession
    # ------------------------------------------------------------------
    is_min = (r[1:-1] < r[:-2]) & (r[1:-1] < r[2:])
    min_indices = np.where(is_min)[0] + 1
    peri_angles = np.array([])

    if len(min_indices) > 1:
        angles = np.unwrap(np.arctan2(y, x))
        peri_angles = angles[min_indices]
        diffs = np.diff(peri_angles)
        precession_measured = float(np.mean(diffs) - 2.0 * math.pi)
    else:
        precession_measured = 0.0

    precession_theory = 6.0 * math.pi * cfg.G * cfg.M / (cfg.c ** 2 * cfg.p)

    log(f"Max Bell violation: {max_violation:.4f}")
    log(f"Violation fraction: {total_violation_fraction:.1%}")

    return BellSimulationResults(
        run_config=run_cfg,
        complexity_power=cp,
        k_rq_ratio=k_rq_ratio,
        r_q_transition=r_q_transition,
        alpha_gr=alpha_gr,
        t=t, x=x, y=y, z=z,
        vx=vx, vy=vy, vz=vz,
        r=r, v2=v2,
        C_Q=C_Q,
        clock_q=clock_q,
        clock_gr=clock_gr,
        E=E, L=L_arr,
        peri_angles=peri_angles,
        precession_measured=precession_measured,
        precession_theory=precession_theory,
        bloch=bloch,
        photons=photons,
        coherence=coherence,
        bell_parameter=bell_parameter,
        entanglement_entropy=entanglement_entropy,
        concurrence=concurrence,
        quantum_discord=quantum_discord,
        bell_violation_regions=violation_regions,
        max_violation=max_violation,
        total_violation_fraction=total_violation_fraction,
    )

# ======================================================================
#  ENTANGLEMENT CONSOLE (Stage 1)
# ======================================================================


class EntanglementConsole:
    """
    Bell state configuration interface with entanglement focus.
    """

    def __init__(self):
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle(
            "⟨ QUANTUM GRAVITY LAB :: BELL STATE EDITION ⟩",
            fontsize=18,
            color=THEME["bell_blue"],
            fontweight="bold",
            y=0.96,
        )

        gs = gridspec.GridSpec(
            3, 3,
            width_ratios=[1, 1, 1],
            height_ratios=[1, 3, 1],
            hspace=0.3,
            wspace=0.3,
        )

        # Header with Bell inequality visualization
        self.ax_header = self.fig.add_subplot(gs[0, :])
        self._draw_bell_header()

        # Left: presets
        self.ax_presets = self.fig.add_subplot(gs[1, 0])
        self.ax_presets.axis("off")

        # Center: orbital params
        self.ax_params = self.fig.add_subplot(gs[1, 1])
        self.ax_params.axis("off")

        # Right: Bell controls
        self.ax_bell = self.fig.add_subplot(gs[1, 2])
        self.ax_bell.axis("off")

        # Bottom: status
        self.ax_status = self.fig.add_subplot(gs[2, :])
        self.ax_status.axis("off")

        self.sliders = {}
        self.status_text = None

        self._build_presets()
        self._build_params()
        self._build_bell_controls()
        self._build_status()

    def _draw_bell_header(self):
        """Visual representation of Bell inequality."""
        self.ax_header.axis("off")

        x = np.linspace(0, 1, 100)
        classical_bound = 0.5 * np.ones_like(x)
        quantum_bound = 0.5 + 0.207 * np.sin(6 * np.pi * x)

        self.ax_header.fill_between(
            x, 0, classical_bound,
            color=THEME["classical"],
            alpha=0.2,
            label="Classical",
        )
        self.ax_header.fill_between(
            x, classical_bound, quantum_bound,
            color=THEME["violation"],
            alpha=0.3,
            label="Bell Violation",
        )
        self.ax_header.plot(x, quantum_bound, color=THEME["bell_purple"], linewidth=2)

        self.ax_header.text(
            0.5, 0.9,
            "CHSH: |⟨A₁B₁⟩ + ⟨A₁B₂⟩ + ⟨A₂B₁⟩ - ⟨A₂B₂⟩| ≤ 2 (classical) vs 2√2 (quantum)",
            transform=self.ax_header.transAxes,
            ha="center",
            fontsize=10,
            color=THEME["fg"],
        )

        self.ax_header.set_xlim(0, 1)
        self.ax_header.set_ylim(0, 1)

    def _build_presets(self):
        presets = [
            ("MAX VIOLATION", RunConfig(p=50, e=0.7, n_orbits=8, n_output=10000)),
            ("DECOHERENCE", RunConfig(p=120, e=0.3, n_orbits=6, n_output=8000)),
            ("SINGULARITY", RunConfig(p=40, e=0.9, n_orbits=10, n_output=12000)),
        ]

        self.ax_presets.text(
            0.5, 0.9,
            "ENTANGLEMENT PROFILES",
            transform=self.ax_presets.transAxes,
            ha="center",
            fontsize=11,
            color=THEME["bell_purple"],
            fontweight="bold",
        )

        self.preset_buttons = []
        for i, (name, cfg) in enumerate(presets):
            ax_btn = self.fig.add_axes(
                [0.08, 0.55 - i * 0.08, 0.20, 0.05],
                facecolor=THEME["panel"],
            )
            btn = Button(ax_btn, name, color=THEME["panel"], hovercolor=THEME["bell_blue"])
            btn.label.set_color(THEME["bell_blue"])
            btn.on_clicked(lambda _evt, c=cfg: self._load_preset(c))
            self.preset_buttons.append(btn)

    def _build_params(self):
        self.ax_params.text(
            0.5, 0.9,
            "ORBITAL PARAMETERS",
            transform=self.ax_params.transAxes,
            ha="center",
            fontsize=11,
            color=THEME["bell_gold"],
            fontweight="bold",
        )

        params = [
            ("p", 30, 200, 60),
            ("e", 0.1, 0.95, 0.6),
            ("n_orbits", 2, 15, 8),
            ("n_output", 4000, 20000, 10000),
        ]

        for i, (name, vmin, vmax, vinit) in enumerate(params):
            ax = self.fig.add_axes(
                [0.38, 0.60 - i * 0.07, 0.20, 0.03],
                facecolor=THEME["panel"],
            )
            s = Slider(ax, name, vmin, vmax, valinit=vinit, color=THEME["bell_gold"])
            self.sliders[name] = s

    def _build_bell_controls(self):
        self.ax_bell.text(
            0.5, 0.9,
            "BELL MEASUREMENTS",
            transform=self.ax_bell.transAxes,
            ha="center",
            fontsize=11,
            color=THEME["violation"],
            fontweight="bold",
        )

        ax_radio = self.fig.add_axes([0.70, 0.45, 0.15, 0.15], facecolor=THEME["bg"])
        self.basis_radio = RadioButtons(
            ax_radio,
            ["XY plane", "XZ plane", "YZ plane"],
            active=0,
        )

        jc_params = [
            ("s_ρ", 0, 2, 1.0),
            ("s_φ", 0, 2, 0.6),
            ("s_v", 0, 2, 0.4),
        ]

        for i, (name, vmin, vmax, vinit) in enumerate(jc_params):
            ax = self.fig.add_axes(
                [0.68, 0.35 - i * 0.05, 0.20, 0.03],
                facecolor=THEME["panel"],
            )
            s = Slider(ax, name, vmin, vmax, valinit=vinit, color=THEME["violation"])
            self.sliders[name] = s

    def _build_status(self):
        ax_launch = self.fig.add_axes(
            [0.40, 0.05, 0.20, 0.05],
            facecolor=THEME["violation"],
        )
        self.btn_launch = Button(ax_launch, "◈ ENTANGLE ◈", color=THEME["violation"])
        self.btn_launch.label.set_color(THEME["bg"])
        self.btn_launch.label.set_fontweight("bold")
        self.btn_launch.on_clicked(self._on_launch)

        self.status_text = self.ax_status.text(
            0.5, 0.3,
            "System ready. Awaiting entanglement.",
            transform=self.ax_status.transAxes,
            ha="center",
            fontsize=10,
            color=THEME["fg"],
            family="monospace",
        )

    def _load_preset(self, cfg: RunConfig):
        self.sliders["p"].set_val(cfg.p)
        self.sliders["e"].set_val(cfg.e)
        self.sliders["n_orbits"].set_val(cfg.n_orbits)
        self.sliders["n_output"].set_val(cfg.n_output)

    def _on_launch(self, _event):
        """
        Launch button:
          - read sliders
          - spawn a NEW Python process that runs Phase 2+3 only
          - close this console window
        """
        self.btn_launch.label.set_text("◈ ENTANGLING... ◈")
        self.status_text.set_text("Launching entangled viewer in a separate process…")
        self.fig.canvas.draw_idle()

        basis_map = {"XY plane": "XY", "XZ plane": "XZ", "YZ plane": "YZ"}

        # Read current UI values
        p = float(self.sliders["p"].val)
        e = float(self.sliders["e"].val)
        n_orbits = int(self.sliders["n_orbits"].val)
        n_output = int(self.sliders["n_output"].val)
        s_rho = float(self.sliders["s_ρ"].val)
        s_phi = float(self.sliders["s_φ"].val)
        s_v = float(self.sliders["s_v"].val)
        basis = basis_map[self.basis_radio.value_selected]

        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--run",
            f"--p={p}",
            f"--e={e}",
            f"--n_orbits={n_orbits}",
            f"--n_output={n_output}",
            f"--s_rho={s_rho}",
            f"--s_phi={s_phi}",
            f"--s_v={s_v}",
            f"--basis={basis}",
        ]

        try:
            subprocess.Popen(cmd)
            self.status_text.set_text(
                "Viewer launched in a new process. This console can now be closed."
            )
            self.fig.canvas.draw_idle()
            # Kill this window / process so the CPU loop from the console is gone
            plt.close(self.fig)
        except Exception as exc:
            self.status_text.set_text(f"Launch failed: {exc}")
            self.fig.canvas.draw_idle()


    def show(self):
        plt.show()

    def run(self):
        """Dark-Matter-style entrypoint."""
        plt.show()    

# ======================================================================
#  BELL CORRELATION VIEWER (Stage 3)
# ======================================================================


class BellCorrelationViewer:
    """
    Quantum correlation dashboard emphasizing Bell violations.

    This version:
      - Uses a time slider + play/pause + speed slider
      - Uses FuncAnimation ONLY to advance the slider
      - Bloch sphere uses preallocated arrow + trail (no per-frame allocations)
    """

    def __init__(self, results: BellSimulationResults):
        self.res = results
        self.i = 0
        self.trail_length = 20  # number of Bloch points kept in the trail
        self.playing = True
        self.speed = 10  # indices per animation step

        self.fig = plt.figure(figsize=(24, 13))
        self.fig.suptitle(
            "◈ BELL STATE QUANTUM CORRELATIONS IN CURVED SPACETIME ◈",
            fontsize=16,
            color=THEME["bell_blue"],
            fontweight="bold",
            y=0.97,
        )

        gs = gridspec.GridSpec(
            3, 4,
            width_ratios=[1, 1, 1.2, 1],
            height_ratios=[1, 1, 1],
            hspace=0.3,
            wspace=0.3,
        )

        self._create_panels(gs)
        self._draw_static()
        self._create_controls()

        # Lightweight animation: just drives the time slider
        self.anim = FuncAnimation(
            self.fig,
            self._animate,
            frames=10_000,
            interval=40,
            blit=False,
        )

    # ------------------------------------------------------------------
    # Panel layout
    # ------------------------------------------------------------------
    def _downsample_idx(self, N: int, max_points: int = 1500) -> np.ndarray:
        """
        Return a set of indices 0..N-1 downsampled to at most max_points.
        Keeps things light so plots don't become solid blocks.
        """
        if N <= max_points:
            return np.arange(N, dtype=int)
        step = max(1, N // max_points)
        return np.arange(0, N, step, dtype=int)

    def _create_panels(self, gs):
        self.ax_bell = self.fig.add_subplot(gs[0, 0])
        self.ax_entropy = self.fig.add_subplot(gs[1, 0])
        self.ax_discord = self.fig.add_subplot(gs[2, 0])

        self.ax_concurrence = self.fig.add_subplot(gs[0, 1])
        self.ax_violation = self.fig.add_subplot(gs[1, 1])
        self.ax_complexity = self.fig.add_subplot(gs[2, 1])

        self.ax_orbit = self.fig.add_subplot(gs[:2, 2])
        self.ax_phase = self.fig.add_subplot(gs[2, 2])

        self.ax_bloch = self.fig.add_subplot(gs[0, 3], projection="3d")
        self.ax_photons = self.fig.add_subplot(gs[1, 3])
        self.ax_coherence = self.fig.add_subplot(gs[2, 3])

        for ax in [
            self.ax_bell, self.ax_entropy, self.ax_discord,
            self.ax_concurrence, self.ax_violation, self.ax_complexity,
            self.ax_phase, self.ax_photons, self.ax_coherence,
        ]:
            ax.grid(True, alpha=0.3)
            ax.set_facecolor(THEME["bg"])

    # ------------------------------------------------------------------
    # Static drawing
    # ------------------------------------------------------------------

    def _draw_static(self):
        t = self.res.t

        # Bell β(t)
        self.ax_bell.plot(t, self.res.bell_parameter, color=THEME["bell_purple"], linewidth=2)
        self.ax_bell.axhline(2.0, color=THEME["classical"], linestyle="--", alpha=0.5, label="Classical")
        self.ax_bell.axhline(2.0 * math.sqrt(2.0), color=THEME["violation"], linestyle="--", alpha=0.5, label="Tsirelson")
        self.ax_bell.fill_between(
            t, 2.0, self.res.bell_parameter,
            where=(self.res.bell_parameter > 2.0),
            color=THEME["violation"], alpha=0.2,
        )
        self.ax_bell.set_ylabel("Bell β")
        self.ax_bell.set_title("CHSH Parameter", color=THEME["bell_purple"])
        self.ax_bell.legend(fontsize=8)

        # Entanglement entropy vs radius
        idx_S = self._downsample_idx(len(self.res.t))
        r_ds = self.res.r[idx_S]
        S_ds = self.res.entanglement_entropy[idx_S]
        beta_ds = self.res.bell_parameter[idx_S]

        self.ax_entropy.scatter(
            r_ds,
            S_ds,
            s=4,
            c=beta_ds,
            cmap="plasma",
            alpha=0.8,
        )
        self.ax_entropy.set_xlabel("r")
        self.ax_entropy.set_ylabel("S(ρ)")
        self.ax_entropy.set_title("Entanglement vs Radius", color=THEME["bell_blue"])

        # Quantum discord vs complexity
        idx_D = self._downsample_idx(len(self.res.t))
        Cq_ds = self.res.C_Q[idx_D]
        D_ds = self.res.quantum_discord[idx_D]
        beta_ds2 = self.res.bell_parameter[idx_D]

        self.ax_discord.scatter(
            Cq_ds,
            D_ds,
            s=4,
            c=beta_ds2,
            cmap="viridis",
            alpha=0.8,
        )
        self.ax_discord.set_xlabel("C_Q")
        self.ax_discord.set_ylabel("D(ρ)")
        self.ax_discord.set_title("Discord vs Complexity", color=THEME["bell_gold"])

        # Concurrence – sparse scatter, coloured by Bell β
        idx_C = self._downsample_idx(len(self.res.t))
        t_C   = self.res.t[idx_C]
        Cc_ds = self.res.concurrence[idx_C]
        beta_C = self.res.bell_parameter[idx_C]

        self.ax_concurrence.scatter(
            t_C,
            Cc_ds,
            s=4,
            c=beta_C,
            cmap="magma",
            alpha=0.8,
        )
        self.ax_concurrence.set_ylabel("C(ρ)")
        self.ax_concurrence.set_title("Concurrence", color=THEME["violation"])

        # Violation regions
        for start, end in self.res.bell_violation_regions:
            self.ax_violation.axvspan(start, end, color=THEME["violation"], alpha=0.3)
        self.ax_violation.plot(t, self.res.bell_parameter - 2.0, color=THEME["bell_purple"], linewidth=2)
        self.ax_violation.axhline(0, color=THEME["fg"], linestyle="-", alpha=0.3)
        self.ax_violation.set_ylabel("β - 2")
        self.ax_violation.set_title(
            f"Violation Regions ({self.res.total_violation_fraction:.1%})",
            color=THEME["violation"],
        )

        # Complexity
        self.ax_complexity.plot(t, self.res.C_Q, color=THEME["bell_gold"], linewidth=2)
        self.ax_complexity.set_ylabel("C_Q")
        self.ax_complexity.set_xlabel("Time")
        self.ax_complexity.set_title("Quantum Complexity", color=THEME["bell_gold"])

        # Orbit coloured by Bell β
        sc = self.ax_orbit.scatter(
            self.res.x, self.res.y,
            c=self.res.bell_parameter,
            s=4,
            cmap="twilight",
            vmin=2.0,
            vmax=2.0 * math.sqrt(2.0),
            alpha=0.8,
        )
        cbar = self.fig.colorbar(sc, ax=self.ax_orbit, fraction=0.046, pad=0.04)
        cbar.set_label("Bell β", fontsize=9)

        violation_mask = self.res.bell_parameter > 2.0
        if np.any(violation_mask):
            self.ax_orbit.scatter(
                self.res.x[violation_mask],
                self.res.y[violation_mask],
                s=1,
                color=THEME["violation"],
                alpha=0.3,
            )

        self.ax_orbit.plot(0, 0, "*", color=THEME["bell_gold"], markersize=12)
        self.ax_orbit.set_aspect("equal")
        self.ax_orbit.set_xlabel("x [GM/c²]")
        self.ax_orbit.set_ylabel("y [GM/c²]")
        self.ax_orbit.set_title("Orbital Trajectory (Bell Correlation Map)", color=THEME["bell_purple"])

        # Phase space
        self.ax_phase.scatter(
            self.res.r,
            np.sqrt(self.res.v2),
            c=self.res.bell_parameter,
            s=2,
            cmap="twilight",
            vmin=2.0,
            vmax=2.0 * math.sqrt(2.0),
            alpha=0.6,
        )
        self.ax_phase.set_xlabel("r")
        self.ax_phase.set_ylabel("|v|")
        self.ax_phase.set_title("Phase Space", color=THEME["fg"])

        # Photon number vs Bell parameter
        idx_n = self._downsample_idx(len(self.res.t))
        beta_n = self.res.bell_parameter[idx_n]
        n_ds = self.res.photons[idx_n]
        r_ds2 = self.res.r[idx_n]

        self.ax_photons.scatter(
            beta_n,
            n_ds,
            s=4,
            c=r_ds2,
            cmap="magma",
            alpha=0.8,
        )
        self.ax_photons.set_xlabel("Bell β")
        self.ax_photons.set_ylabel("⟨n⟩")
        self.ax_photons.set_title("Photons vs Bell Parameter", color=THEME["bell_blue"])

        # Coherence – sparse scatter vs time, coloured by C_Q
        idx_coh = self._downsample_idx(len(self.res.t))
        t_coh   = self.res.t[idx_coh]
        coh_ds  = self.res.coherence[idx_coh]
        Cq_coh  = self.res.C_Q[idx_coh]

        self.ax_coherence.scatter(
            t_coh,
            coh_ds,
            s=4,
            c=Cq_coh,
            cmap="plasma",
            alpha=0.8,
        )
        self.ax_coherence.set_ylabel("|ρ₀₁|")
        self.ax_coherence.set_xlabel("Time")
        self.ax_coherence.set_title("Coherence", color=THEME["bell_purple"])

        # Bloch sphere
        self._setup_bloch()

        # Cursors on time-series panels
        self.cursors = []
        for ax in [
            self.ax_bell,
            self.ax_concurrence,
            self.ax_violation,
            self.ax_complexity,
            self.ax_coherence,
        ]:
            line = ax.axvline(t[0], color=THEME["fg"], linestyle=":", alpha=0.5)
            self.cursors.append(line)

        # Orbit marker
        self.orbit_marker, = self.ax_orbit.plot(
            [], [], "o",
            color=THEME["bell_blue"],
            markersize=8,
            markeredgecolor=THEME["fg"],
        )

    # ------------------------------------------------------------------
    # Bloch sphere
    # ------------------------------------------------------------------
    def _setup_bloch(self):
        """Setup Bloch sphere visualization with preallocated arrow + trail."""
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        # Bright cyan sphere on black
        self.ax_bloch.plot_wireframe(
            x, y, z,
            color="#373d3d",
            linewidth=0.6,
            alpha=0.9,
        )

        # Keep a cube-ish view but no visible frame
        self.ax_bloch.set_xlim([-1.2, 1.2])
        self.ax_bloch.set_ylim([-1.2, 1.2])
        self.ax_bloch.set_zlim([-1.2, 1.2])
        self.ax_bloch.set_box_aspect([1, 1, 1])
        self.ax_bloch.set_title("Qubit State", color=THEME["bell_purple"])

        # Remove ticks, labels, and the 3D box frame
        self.ax_bloch.set_xticks([])
        self.ax_bloch.set_yticks([])
        self.ax_bloch.set_zticks([])
        self.ax_bloch.set_axis_off()

        # Arrow (bright purple) – initialize pointing up
        self.bloch_arrow = Line3D(
            [0.0, 0.0], [0.0, 0.0], [0.0, 1.1],
            linewidth=2.5,
            color="#ff66ff",
        )
        self.ax_bloch.add_line(self.bloch_arrow)

        # Trail (cyan)
        self.bloch_trail = Line3D(
            [], [], [],
            linewidth=1.5,
            alpha=0.7,
            color="#ff66ff",
        )
        self.ax_bloch.add_line(self.bloch_trail)

        if getattr(self.res, "bloch", None) is not None and len(self.res.bloch) > 0:
            self._update_bloch(0)


    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _create_controls(self):
        N = len(self.res.t)

        # Time slider
        ax_time = plt.axes([0.20, 0.02, 0.55, 0.02], facecolor=THEME["panel"])
        self.time_slider = Slider(
            ax_time,
            "",
            0,
            max(0, N - 1),
            valinit=0,
            valstep=1,
            color=THEME["bell_blue"],
        )
        self.time_slider.on_changed(self._on_seek)

        # Play / pause button
        ax_play = plt.axes([0.08, 0.02, 0.08, 0.035], facecolor=THEME["panel"])
        self.btn_play = Button(
            ax_play,
            "❚❚",
            color=THEME["panel"],
            hovercolor=THEME["bell_blue"],
        )
        self.btn_play.label.set_color(THEME["bell_blue"])
        self.btn_play.on_clicked(self._toggle_play)

        # Speed slider
        ax_speed = plt.axes([0.80, 0.02, 0.12, 0.02], facecolor=THEME["panel"])
        self.speed_slider = Slider(
            ax_speed,
            "speed",
            1,
            100,
            valinit=self.speed,
            valstep=1,
            color=THEME["bell_gold"],
        )
        self.speed_slider.on_changed(lambda v: setattr(self, "speed", int(v)))

        info = (
            f"Max Bell: {self.res.max_violation:.4f} | "
            f"Violation: {self.res.total_violation_fraction:.1%} | "
            f"Precession: {self.res.precession_measured:.6f} rad/orbit"
        )
        self.fig.text(
            0.5, 0.005,
            info,
            ha="center",
            fontsize=10,
            color=THEME["fg"],
            family="monospace",
        )

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _on_seek(self, val):
        """Seek to a time index and update cursors + orbit marker + Bloch sphere."""
        i = int(val)
        if len(self.res.t) == 0:
            return

        i = max(0, min(i, len(self.res.t) - 1))
        self.i = i
        t_i = self.res.t[i]

        # Update vertical cursors
        for cursor in getattr(self, "cursors", []):
            cursor.set_xdata([t_i, t_i])

        # Update orbit marker
        xi = float(self.res.x[i])
        yi = float(self.res.y[i])
        if np.all(np.isfinite([xi, yi])):
            self.orbit_marker.set_data([xi], [yi])

        # Update Bloch sphere safely
        self._update_bloch(i)

        self.fig.canvas.draw_idle()

    def _update_bloch(self, idx: int):
        """
        Safe Bloch update:
        - clamps index
        - ignores non-finite data
        - draws arrow & trail slightly outside the sphere
        """
        bloch = getattr(self.res, "bloch", None)
        if bloch is None or len(bloch) == 0:
            return

        N = len(bloch)
        idx = max(0, min(idx, N - 1))

        sx, sy, sz = bloch[idx]
        if not np.all(np.isfinite([sx, sy, sz])):
            return

        # Push the visible geometry slightly outside radius 1
        R_arrow = 1.08   # how far outside the sphere the arrow tip lives
        R_trail = 1.02   # trail hugs the surface a bit more closely

        try:
            # Arrow
            self.bloch_arrow.set_data_3d(
                [0.0, R_arrow * sx],
                [0.0, R_arrow * sy],
                [0.0, R_arrow * sz],
            )

            # Trail
            if idx > 0:
                start = max(0, idx - self.trail_length)
                trail = bloch[start: idx + 1]
                xs = R_trail * trail[:, 0]
                ys = R_trail * trail[:, 1]
                zs = R_trail * trail[:, 2]
            else:
                xs = ys = zs = []

            self.bloch_trail.set_data_3d(xs, ys, zs)

        except Exception as exc:
            print(f"[BLOCH] update failed at idx={idx}: {exc}")
            return

    def _toggle_play(self, _event):
        self.playing = not self.playing
        self.btn_play.label.set_text("▶" if not self.playing else "❚❚")

    def _animate(self, _frame):
        """Animation callback: advance the slider if playing, with hard guards."""
        if not self.playing or len(self.res.t) == 0:
            return []

        try:
            self.i = (self.i + self.speed) % len(self.res.t)
            self.time_slider.set_val(self.i)
        except Exception as exc:
            print(f"[ANIM] animation stopped at i={self.i}: {exc}")
            self.playing = False
            try:
                self.btn_play.label.set_text("▶")
            except Exception:
                pass

        return []

    # ------------------------------------------------------------------
    # Show
    # ------------------------------------------------------------------

    def show(self):
        # Show this figure inside an already-running Qt/Matplotlib loop
        self.fig.show()

# ======================================================================
#  MAIN ENTRY
# ======================================================================

if __name__ == "__main__":
    print("╔" + "═" * 58 + "╗")
    print("║   QUANTUM GRAVITY LAB :: BELL STATE EDITION            ║")
    print("║   Exploring quantum correlations in curved spacetime    ║")
    print("╚" + "═" * 58 + "╝")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run Phase 2+3 only (no console)")
    parser.add_argument("--p", type=float, default=60.0)
    parser.add_argument("--e", type=float, default=0.6)
    parser.add_argument("--n_orbits", type=int, default=8)
    parser.add_argument("--n_output", type=int, default=10000)
    parser.add_argument("--s_rho", type=float, default=1.0)
    parser.add_argument("--s_phi", type=float, default=0.6)
    parser.add_argument("--s_v", type=float, default=0.4)
    parser.add_argument("--basis", type=str, default="XY")  # "XY", "XZ", or "YZ"
    args = parser.parse_args()

    if args.run:
        # PHASE 2+3 ONLY — this is the path that already works for you
        run_cfg = RunConfig(
            p=args.p,
            e=args.e,
            n_orbits=args.n_orbits,
            n_output=args.n_output,
            s_rho=args.s_rho,
            s_phi=args.s_phi,
            s_v=args.s_v,
            with_jc=True,
            use_quantum_time=True,
            track_bell=True,
            measurement_basis=args.basis,
        )

        results = run_entangled_simulation(run_cfg, status_cb=None)
        viewer = BellCorrelationViewer(results)
        viewer.show()
        plt.show()
    else:
        # Default: console UI
        ui = EntanglementConsole()
        ui.run()


