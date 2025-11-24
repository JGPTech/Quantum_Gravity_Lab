#!/usr/bin/env python3
"""
Quantum Gravity Lab — Offline Edition
=====================================

Two-stage interface:

1) RUN SELECTION / LOADING SCREEN
   - Choose preset or custom (p, e, n_orbits, n_output)
   - On "Run Simulation":
       * Look up complexity_power in calibration CSV, or
       * Call calibrate.calibrate_complexity_power(p, e, ...) if missing
       * Save new calibration row back to CSV
       * Run unified_sandbox_core.run_unified_trajectory once (offline)
       * Build a rich `results` dict (orbit, clocks, JC, metrics)

2) RESULTS VIEWER
   - Replays the precomputed run with a time cursor:
       * Left: GR metrics (precession, clocks, energy, L)
       * Center: orbit, complexity, clock error
       * Right: Bloch sphere, photons, coherence, JC params
   - Play/pause + speed slider + time scrub slider

Designed so adding new graphs is "plug and play": just consume `results`.
"""

import math
import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.linalg import expm

from unified_sandbox_core import (
    Config,
    canonical_periapsis_ic,
    run_unified_trajectory,
    compute_quantum_clock_rate,
)

# IMPORTANT: your calibration module
from calibrate import calibrate_complexity_power


# ======================================================================
#  Data structures
# ======================================================================

CALIBRATION_CSV = "complexity_power_grid_scan.csv"


@dataclass
class RunConfig:
    """Parameters chosen on the loading screen."""
    p: float
    e: float
    n_orbits: int
    n_output: int
    s_rho: float = 1.0
    s_phi: float = 0.6
    s_v: float = 0.4
    with_jc: bool = True
    use_quantum_time: bool = True


@dataclass
class SimulationResults:
    """Container for everything the viewer needs."""
    # core config / run setup
    run_config: RunConfig
    complexity_power: float
    k_rq_ratio: float | None
    r_q_transition: float
    alpha_gr: float

    # orbit / kinematics
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    r: np.ndarray
    v2: np.ndarray

    # clocks / complexity
    C_Q: np.ndarray
    clock_q: np.ndarray
    clock_gr: np.ndarray

    # energy / angular momentum
    E: np.ndarray
    L: np.ndarray

    # GR metrics
    peri_angles: np.ndarray
    precession_measured: float
    precession_theory: float
    energy_variation: float
    clock_max_dilation: float
    clock_max_dilation_gr: float

    # accuracy scores
    accuracy_scores: dict

    # JC / quantum
    bloch: np.ndarray           # (N, 3)
    photons: np.ndarray         # (N,)
    coherence: np.ndarray       # (N,)
    omega_c: np.ndarray         # (N,)
    omega_a: np.ndarray         # (N,)


# ======================================================================
#  Calibration helpers
# ======================================================================

def load_calibration_df() -> pd.DataFrame:
    """Load calibration CSV if it exists, else return empty DataFrame
    with all expected columns (including JC couplings)."""
    cols = [
        "p", "e",
        "n_orbits", "n_output",
        "s_rho", "s_phi", "s_v",
        "ok", "complexity_power",
        "rel_error", "measured_shift", "theory_shift", "n_peri",
        "k_rq_ratio", "r_q_transition",
    ]

    if os.path.exists(CALIBRATION_CSV):
        df = pd.read_csv(CALIBRATION_CSV)
        # Ensure any missing columns are added as NaN so older files still load
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[cols]

    return pd.DataFrame(columns=cols)


def save_calibration_df(df: pd.DataFrame) -> None:
    df.to_csv(CALIBRATION_CSV, index=False)


def find_calibrated_row(
    df: pd.DataFrame,
    p: float,
    e: float,
    n_orbits: int,
    n_output: int,
    s_rho: float,
    s_phi: float,
    s_v: float,
) -> pd.Series | None:
    """Return best existing calibration row for the full slider state, or None."""
    if df.empty:
        return None

    mask = (
        np.isclose(df["p"], p)
        & np.isclose(df["e"], e)
        & (df["n_orbits"] == int(n_orbits))
        & (df["n_output"] == int(n_output))
        & np.isclose(df["s_rho"], s_rho)
        & np.isclose(df["s_phi"], s_phi)
        & np.isclose(df["s_v"], s_v)
    )

    rows = df[mask]
    if rows.empty:
        return None

    idx = rows["rel_error"].abs().idxmin()
    return rows.loc[idx]


def get_or_calibrate_complexity_power(
    p: float,
    e: float,
    n_orbits: int,
    n_output: int,
    s_rho: float,
    s_phi: float,
    s_v: float,
    df_calib: pd.DataFrame,
) -> tuple[float, pd.DataFrame, dict]:
    """
    Look up complexity_power for the full slider state.
    If not present, run calibrate_complexity_power with the same
    (p, e, n_orbits, n_output), tag it with (s_rho, s_phi, s_v), and save.

    Returns:
        complexity_power, updated_df_calib, calib_record_dict
    """
    # 1) Try lookup
    row = find_calibrated_row(df_calib, p, e, n_orbits, n_output, s_rho, s_phi, s_v)
    if row is not None and bool(row.get("ok", True)):
        cp = float(row["complexity_power"])
        record = row.to_dict()
        return cp, df_calib, record

    # 2) Need to calibrate
    print(
        "\n=== No calibration found for slider state ===\n"
        f"    p={p:.3f}, e={e:.3f}, "
        f"n_orbits={n_orbits}, n_output={n_output}, "
        f"s_rho={s_rho:.3f}, s_phi={s_phi:.3f}, s_v={s_v:.3f}"
        "\n    → running calibrate_complexity_power ==="
    )

    res = calibrate_complexity_power(
        p=p,
        e=e,
        n_orbits=n_orbits,
        n_output=n_output,
        coarse_range=(0.2, 2.5),
        coarse_steps=9,
        fine_half_width=0.20,
        fine_steps=40,
    )

    # Build flat record compatible with CSV, tagged with JC couplings
    record = {
        "p": p,
        "e": e,
        "n_orbits": n_orbits,
        "n_output": n_output,
        "s_rho": float(s_rho),
        "s_phi": float(s_phi),
        "s_v": float(s_v),
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
    print(f"Stored new calibration: complexity_power = {cp:.6f}")
    return cp, df_calib, record


# ======================================================================
#  JC helpers (offline, reused in viewer)
# ======================================================================
def remap_bloch_to_full_sphere(bloch_vecs: np.ndarray) -> np.ndarray:
    """
    Stretch the polar angle range of the Bloch trajectory so that
    the smallest theta goes to the north pole (z=+1) and the largest
    theta goes to the south pole (z=-1), using the full sphere.

    bloch_vecs: (N, 3) array of [sx, sy, sz].
    Returns: (N, 3) array of remapped [sx, sy, sz] for visualization.
    """
    if bloch_vecs.size == 0:
        return bloch_vecs

    sx = bloch_vecs[:, 0]
    sy = bloch_vecs[:, 1]
    sz = bloch_vecs[:, 2]

    r = np.sqrt(sx**2 + sy**2 + sz**2)
    r[r == 0] = 1.0

    # Original spherical angles
    theta = np.arccos(np.clip(sz / r, -1.0, 1.0))  # [0, π]
    phi = np.arctan2(sy, sx)                       # [-π, π]

    theta_min = float(theta.min())
    theta_max = float(theta.max())

    # If there's no range, just return a unit-normalized copy
    if not np.isfinite(theta_min) or not np.isfinite(theta_max) or theta_max <= theta_min:
        norm = np.maximum(r, 1e-12)
        return np.stack([sx / norm, sy / norm, sz / norm], axis=1)

    # Linearly map [theta_min, theta_max] → [0, π]
    theta_norm = (theta - theta_min) / (theta_max - theta_min)
    theta_stretched = theta_norm * np.pi

    # Rebuild unit vectors on the sphere
    sx_new = np.sin(theta_stretched) * np.cos(phi)
    sy_new = np.sin(theta_stretched) * np.sin(phi)
    sz_new = np.cos(theta_stretched)

    return np.stack([sx_new, sy_new, sz_new], axis=1)

def initialize_jc_state(dim: int, n_cavity: int) -> np.ndarray:
    """Coherent-like mixed JC initial state."""
    rho = np.zeros((dim, dim), dtype=np.complex128)
    if dim < 6:
        return rho

    alpha = 1.5
    for n in range(min(10, n_cavity)):
        prob_n = (
            np.exp(-abs(alpha) ** 2)
            * (abs(alpha) ** (2 * n))
            / math.factorial(n)
        )
        idx_g = 2 * n
        idx_e = idx_g + 1
        if idx_e < dim:
            rho[idx_g, idx_g] += 0.5 * prob_n
            rho[idx_e, idx_e] += 0.5 * prob_n
            if n < 5:
                rho[idx_g, idx_e] += 0.2 * prob_n * np.exp(1j * n * 0.5)
                rho[idx_e, idx_g] += 0.2 * prob_n * np.exp(-1j * n * 0.5)

    tr = np.trace(rho)
    if abs(tr) > 1e-12:
        rho /= tr
    return rho


def jc_params_from_classical(
    r: float,
    v_mag: float,
    G: float,
    M: float,
    jc_rho_ref: float,
    jc_v_ref: float,
    jc_phi_ref: float,
    omega_cavity_0: float,
    omega_atom_0: float,
    g_0: float,
    s_rho: float,
    s_phi: float,
    s_v: float,
) -> dict:
    """Map orbital state → JC parameters."""
    r_safe = max(float(r), 1e-6)
    rho_local = 1.0 / (r_safe ** 2 + 1e-6)
    phi_local = -G * M / r_safe

    rho_hat = rho_local / (jc_rho_ref + 1e-12)
    v_hat = v_mag / (jc_v_ref + 1e-12)
    phi_hat = phi_local / (abs(jc_phi_ref) + 1e-12)

    omega_c = omega_cavity_0 * (1.0 + s_rho * rho_hat)
    omega_a = omega_atom_0 * (1.0 + s_phi * phi_hat)
    g_coup = g_0 * (1.0 + s_v * v_hat)

    return {
        "omega_c": float(omega_c),
        "omega_a": float(omega_a),
        "g": float(g_coup),
    }


def build_jc_hamiltonian(params: dict, dim: int, n_cavity: int) -> np.ndarray:
    """Standard JC Hamiltonian in truncated Fock space."""
    H = np.zeros((dim, dim), dtype=np.complex128)
    omega_c = params["omega_c"]
    omega_a = params["omega_a"]
    g = params["g"]

    for n in range(n_cavity):
        idx_g = 2 * n
        idx_e = idx_g + 1
        if idx_e < dim:
            H[idx_g, idx_g] += omega_c * n - 0.5 * omega_a
            H[idx_e, idx_e] += omega_c * n + 0.5 * omega_a

    for n in range(n_cavity - 1):
        idx_e_n = 2 * n + 1
        idx_g_np1 = 2 * (n + 1)
        if idx_g_np1 < dim:
            val = g * np.sqrt(n + 1.0)
            H[idx_g_np1, idx_e_n] += val
            H[idx_e_n, idx_g_np1] += val

    return H


def extract_bloch_vector(rho: np.ndarray, n_cavity: int) -> tuple[float, float, float]:
    """Extract (sx, sy, sz) Bloch vector by tracing over photon number."""
    dim = rho.shape[0]
    sigma_x = 0.0
    sigma_y = 0.0
    sigma_z = 0.0
    total_weight = 0.0

    for n in range(n_cavity):
        idx_g = 2 * n
        idx_e = idx_g + 1
        if idx_e < dim:
            p_g = rho[idx_g, idx_g].real
            p_e = rho[idx_e, idx_e].real
            coh = rho[idx_e, idx_g]
            weight = p_g + p_e
            if weight > 1e-10:
                sigma_x += weight * 2.0 * np.real(coh)
                sigma_y += weight * 2.0 * np.imag(coh)
                sigma_z += weight * (p_e - p_g)
                total_weight += weight

    if total_weight > 0:
        sigma_x /= total_weight
        sigma_y /= total_weight
        sigma_z /= total_weight

    norm = math.sqrt(sigma_x ** 2 + sigma_y ** 2 + sigma_z ** 2)
    if norm > 0:
        return sigma_x / norm, sigma_y / norm, sigma_z / norm
    else:
        return 0.0, 0.0, 1.0


def jc_expectations(rho: np.ndarray, n_cavity: int) -> tuple[float, float]:
    """⟨n⟩ and total coherence magnitude."""
    dim = rho.shape[0]
    n_photons = 0.0
    coherence_mag = 0.0
    for n in range(n_cavity):
        idx_g = 2 * n
        idx_e = idx_g + 1
        if idx_e < dim:
            p_g = rho[idx_g, idx_g].real
            p_e = rho[idx_e, idx_e].real
            n_photons += n * (p_g + p_e)
            coh = rho[idx_e, idx_g]
            coherence_mag += abs(coh)
    return n_photons, coherence_mag


# ======================================================================
#  Offline simulation runner
# ======================================================================

def schwarzschild_clock(r: float, G: float, M: float, c: float) -> float:
    """Schwarzschild time dilation dτ/dt for a static observer."""
    r_g = G * M / (c ** 2)
    x = 1.0 - 2.0 * r_g / max(float(r), 1e-12)
    if x <= 0.0:
        return 0.0
    return float(math.sqrt(x))


def compute_gr_metrics_and_accuracy(
    cfg: Config,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    C_Q: np.ndarray,
    clock_q: np.ndarray,
    clock_gr: np.ndarray,
) -> tuple[dict, dict]:
    """
    Compute GR metrics (precession, energy variation, clock dilation) and
    human-friendly accuracy scores.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    v2 = vx**2 + vy**2 + vz**2

    # Periapsis detection
    is_min = (r[1:-1] < r[:-2]) & (r[1:-1] < r[2:])
    min_indices = np.where(is_min)[0] + 1

    gr_metrics = {}
    peri_angles = np.array([])

    if len(min_indices) > 1:
        angles_all = np.unwrap(np.arctan2(y, x))
        peri_angles = angles_all[min_indices]

        diffs = np.diff(peri_angles)
        avg_shift_total = float(np.mean(diffs))
        avg_shift_excess = avg_shift_total - 2.0 * math.pi

        gr_metrics["precession_measured"] = avg_shift_excess
        gr_metrics["precession_theory"] = (
            6.0 * math.pi * cfg.G * cfg.M / (cfg.c**2 * cfg.p)
        )

    # Energy variation
    E = 0.5 * v2 - cfg.G * cfg.M / np.maximum(r, 1e-8)
    if np.mean(E) != 0.0:
        energy_variation = float(np.std(E) / abs(np.mean(E)))
    else:
        energy_variation = float(np.std(E))
    gr_metrics["energy_variation"] = energy_variation

    # Time dilation extrema
    if len(clock_q) == len(clock_gr) and len(clock_q) > 0:
        gr_metrics["max_dilation"] = 1.0 - float(np.min(clock_q))
        gr_metrics["max_dilation_gr"] = 1.0 - float(np.min(clock_gr))

    # Accuracy scores
    scores: dict[str, float] = {}

    # Precession accuracy
    if "precession_measured" in gr_metrics and "precession_theory" in gr_metrics:
        measured = gr_metrics["precession_measured"]
        theory = gr_metrics["precession_theory"]
        if abs(theory) > 1e-10:
            err = abs((measured - theory) / theory)
            if err < 0.01:
                scores["precession"] = 100.0
            elif err < 1.0:
                scores["precession"] = max(0.0, 100.0 * (1.0 - err))
            else:
                scores["precession"] = 0.0
        else:
            scores["precession"] = 0.0
    else:
        scores["precession"] = 0.0

    # Clock accuracy (mean absolute deviation from Schwarzschild)
    if len(clock_q) == len(clock_gr) and len(clock_q) > 0:
        clock_error = float(np.mean(np.abs(clock_q - clock_gr)))
        scores["clock"] = max(0.0, 100.0 * (1.0 - clock_error))
    else:
        scores["clock"] = 0.0

    # Energy conservation
    ev = gr_metrics.get("energy_variation", 1.0)
    scores["energy"] = max(0.0, 100.0 * (1.0 - ev))

    # Overall
    scores["overall"] = float(np.mean(list(scores.values()))) if scores else 0.0

    return gr_metrics | {"peri_angles": peri_angles}, scores


def run_offline_simulation(run_cfg: RunConfig, status_cb=None) -> SimulationResults:
    """
    Full offline pipeline:
      - Load/update calibration grid
      - Configure Config for (p,e)
      - Run unified trajectory (with JC + quantum time)
      - Evolve JC along the trajectory
      - Compute GR metrics + accuracy
      - Package everything into SimulationResults
    """
    def set_status(msg: str):
        if status_cb is not None:
            status_cb(msg)
        print(msg)

    # 1) Calibration grid
    set_status("Loading calibration grid...")
    df_calib = load_calibration_df()

    # 2) Get or calibrate complexity_power
    cp, df_calib, calib_record = get_or_calibrate_complexity_power(
        p=run_cfg.p,
        e=run_cfg.e,
        n_orbits=run_cfg.n_orbits,
        n_output=run_cfg.n_output,
        s_rho=run_cfg.s_rho,
        s_phi=run_cfg.s_phi,
        s_v=run_cfg.s_v,
        df_calib=df_calib,
    )

    # 3) Build Config for this run
    set_status("Configuring unified sandbox for selected orbit...")
    cfg = Config()
    cfg.p = float(run_cfg.p)
    cfg.e = float(run_cfg.e)
    cfg.complexity_power = float(cp)

    # Dynamic k_rq if enabled
    k_rq_ratio = None
    if getattr(cfg, "k_rq_ratio_mode", "fixed") == "dynamic":
        from unified_sandbox_core import k_rq_from_pe
        cfg.k_rq_ratio = k_rq_from_pe(cfg.p, cfg.e, default=cfg.k_rq_ratio_default)
        k_rq_ratio = float(cfg.k_rq_ratio)
    else:
        k_rq_ratio = getattr(cfg, "k_rq_ratio", None)

    # GR coupling & transition radius
    L_sq = cfg.G * cfg.M * cfg.p
    cfg.alpha_gr = (3.0 * cfg.G * cfg.M * L_sq) / (cfg.c**2)
    if hasattr(cfg, "update_r_q_transition_from_orbit"):
        cfg.update_r_q_transition_from_orbit()
    if hasattr(cfg, "setup_quantum_sensor"):
        cfg.setup_quantum_sensor()

    alpha_gr = float(cfg.alpha_gr)
    r_q_transition = float(getattr(cfg, "r_q_transition", np.nan))

    # 4) Time span
    a = cfg.p / (1.0 - cfg.e**2)
    T_orb = 2.0 * math.pi * math.sqrt(a**3 / (cfg.G * cfg.M))
    t_max = run_cfg.n_orbits * T_orb

    # 5) Initial conditions
    r0, v0 = canonical_periapsis_ic(cfg)

    # 6) Unified trajectory
    set_status("Running unified trajectory (this may take a bit)...")
    df = run_unified_trajectory(
        cfg,
        r0,
        v0,
        t_max,
        n_output=run_cfg.n_output,
        with_jc=run_cfg.with_jc,
        use_quantum_time=run_cfg.use_quantum_time,
    )

    # --- NEW DEBUG ---
    t = df["t"].values
    print("\n[DEBUG] unified trajectory:")
    print(f"  requested t_max   = {t_max:.6e}")
    print(f"  achieved t_final  = {t[-1]:.6e}")
    print(f"  requested n_output = {run_cfg.n_output}")
    print(f"  achieved N         = {len(t)}\n")


    # Extract arrays
    t = df["t"].values
    x = df["x"].values
    y = df["y"].values
    z = df["z"].values
    vx = df["vx"].values
    vy = df["vy"].values
    vz = df["vz"].values
    r = np.sqrt(x**2 + y**2 + z**2)
    v2 = vx**2 + vy**2 + vz**2

    # Complexity & clocks
    set_status("Computing complexity & clocks...")
    C_Q_list = []
    clock_q_list = []
    clock_gr_list = []
    for ri in r:
        _, C_Q_i, clock_i = compute_quantum_clock_rate(ri, cfg)
        C_Q_list.append(C_Q_i)
        clock_q_list.append(clock_i)
        clock_gr_list.append(schwarzschild_clock(ri, cfg.G, cfg.M, cfg.c))

    C_Q = np.array(C_Q_list)
    clock_q = np.array(clock_q_list)
    clock_gr = np.array(clock_gr_list)

    # Energy & angular momentum
    E = 0.5 * v2 - cfg.G * cfg.M / np.maximum(r, 1e-8)
    L_arr = np.zeros_like(r)
    for i in range(len(r)):
        L_vec = np.cross(
            np.array([x[i], y[i], z[i]]),
            np.array([vx[i], vy[i], vz[i]]),
        )
        L_arr[i] = np.linalg.norm(L_vec)

    # 7) JC evolution along trajectory
    set_status("Evolving JC system along trajectory...")
    omega_cavity_0 = 2.0 * math.pi * 1.0
    omega_atom_0 = 2.0 * math.pi * 1.0
    g_0 = 0.09 * 2.0 * math.pi
    n_cavity = 20

    # Geometric coupling strengths from run_config
    s_rho = run_cfg.s_rho
    s_phi = run_cfg.s_phi
    s_v = run_cfg.s_v

    # Reference scales at periapsis
    r_peri = cfg.p / (1.0 + cfg.e)
    jc_rho_ref = 1.0 / (r_peri**2 + 1e-6)
    jc_v_ref = math.sqrt(cfg.G / cfg.p) * (1.0 + cfg.e)
    jc_phi_ref = -cfg.G * cfg.M / max(r_peri, 1e-6)

    dim = 2 * n_cavity
    rho = initialize_jc_state(dim, n_cavity)

    bloch_vecs = []
    photons = []
    coherences = []
    omega_c_hist = []
    omega_a_hist = []

    if len(t) > 1:
        dt = float(np.mean(np.diff(t)))
    else:
        dt = 0.0

    for i in range(len(t)):
        ri = r[i]
        vmag = math.sqrt(v2[i])

        jc_params = jc_params_from_classical(
            ri,
            vmag,
            cfg.G,
            cfg.M,
            jc_rho_ref,
            jc_v_ref,
            jc_phi_ref,
            omega_cavity_0,
            omega_atom_0,
            g_0,
            s_rho,
            s_phi,
            s_v,
        )
        omega_c_hist.append(jc_params["omega_c"] / omega_cavity_0)
        omega_a_hist.append(jc_params["omega_a"] / omega_atom_0)

        if dt != 0.0:
            C_Q_i = C_Q[i]
            H = build_jc_hamiltonian(jc_params, dim, n_cavity)
            dt_quantum = dt * (1.0 + 0.3 * C_Q_i)
            U = expm(-1j * H * dt_quantum)
            rho = U @ rho @ U.conj().T

        sx, sy, sz = extract_bloch_vector(rho, n_cavity)
        bloch_vecs.append([sx, sy, sz])

        n_phot, coh = jc_expectations(rho, n_cavity)
        photons.append(n_phot)
        coherences.append(coh)

    bloch_vecs = np.array(bloch_vecs)
    photons = np.array(photons)
    coherences = np.array(coherences)
    omega_c_hist = np.array(omega_c_hist)
    omega_a_hist = np.array(omega_a_hist)

    # 8) GR metrics + accuracy
    set_status("Computing GR metrics and accuracy...")
    gr_metrics, accuracy_scores = compute_gr_metrics_and_accuracy(
        cfg, t, x, y, z, vx, vy, vz, C_Q, clock_q, clock_gr
    )

    peri_angles = gr_metrics.get("peri_angles", np.array([]))
    precession_measured = float(gr_metrics.get("precession_measured", 0.0))
    precession_theory = float(gr_metrics.get("precession_theory", 0.0))
    energy_variation = float(gr_metrics.get("energy_variation", 0.0))
    clock_max_dilation = float(gr_metrics.get("max_dilation", 0.0))
    clock_max_dilation_gr = float(gr_metrics.get("max_dilation_gr", 0.0))

    set_status("Simulation complete. Launching viewer...")

    return SimulationResults(
        run_config=run_cfg,
        complexity_power=cp,
        k_rq_ratio=k_rq_ratio,
        r_q_transition=r_q_transition,
        alpha_gr=alpha_gr,
        t=t,
        x=x,
        y=y,
        z=z,
        vx=vx,
        vy=vy,
        vz=vz,
        r=r,
        v2=v2,
        C_Q=C_Q,
        clock_q=clock_q,
        clock_gr=clock_gr,
        E=E,
        L=L_arr,
        peri_angles=peri_angles,
        precession_measured=precession_measured,
        precession_theory=precession_theory,
        energy_variation=energy_variation,
        clock_max_dilation=clock_max_dilation,
        clock_max_dilation_gr=clock_max_dilation_gr,
        accuracy_scores=accuracy_scores,
        bloch=bloch_vecs,
        photons=photons,
        coherence=coherences,
        omega_c=omega_c_hist,
        omega_a=omega_a_hist,
    )


# ======================================================================
#  RUN SELECTION / LOADING SCREEN
# ======================================================================

class RunSelectionUI:
    """
    Retro-style loading screen:
      - Left: preset run buttons
      - Right: custom sliders for p, e, n_orbits, n_output, JC couplings
      - Bottom: Run Simulation button + status text
    """

    def __init__(self):
        self.fig = plt.figure(figsize=(12, 7))
        self.fig.suptitle(
            "QUANTUM GRAVITY LAB — OFFLINE EDITION",
            fontsize=16,
            fontweight="bold",
            y=0.95,
        )

        gs = gridspec.GridSpec(
            2, 2,
            height_ratios=[3, 1],
            width_ratios=[1, 1],
            left=0.05,
            right=0.95,
            top=0.90,
            bottom=0.10,
            wspace=0.25,
            hspace=0.30,
        )

        # LEFT TOP: presets
        self.ax_presets = self.fig.add_subplot(gs[0, 0])
        self.ax_presets.axis("off")

        # RIGHT TOP: custom sliders
        self.ax_custom = self.fig.add_subplot(gs[0, 1])
        self.ax_custom.axis("off")

        # BOTTOM: run controls + status
        self.ax_run = self.fig.add_subplot(gs[1, :])
        self.ax_run.axis("off")

        # UI elements
        self.preset_buttons = []
        self.selected_preset = None
        self.custom_sliders = {}
        self.run_button = None
        self.status_text = None

        self._build_presets()
        self._build_custom_controls()
        self._build_run_controls()

    # Presets -----------------------------------------------------------

    def _build_presets(self):
        """
        Preset buttons (simple text/buttons).
        You can add more presets easily here.
        """
        # Define a few nice presets (tweak as you like)
        self.presets = [
            ("Mercury-like", RunConfig(p=80.0, e=0.206, n_orbits=8, n_output=8000)),
            ("Mild Eccentric", RunConfig(p=100.0, e=0.4, n_orbits=6, n_output=6000)),
            ("High-e Brink", RunConfig(p=60.0, e=0.8, n_orbits=6, n_output=8000)),
            ("Random from grid", None),  # special behavior
        ]

        # Layout buttons vertically
        y0 = 0.75
        dy = 0.15
        for i, (label, rc) in enumerate(self.presets):
            ax_btn = self.fig.add_axes([0.10, 0.60 - i * 0.08, 0.20, 0.05])
            btn = Button(ax_btn, label)
            btn.on_clicked(self._make_preset_callback(i))
            self.preset_buttons.append(btn)

        # Helpful text
        self.ax_presets.text(
            0.1, 0.65,
            "Tip: presets load sensible (p,e,n_orbits).\n"
            "You can still tweak custom sliders before running.",
            transform=self.ax_presets.transAxes,
            fontsize=9,
            va="bottom",
        )

    def _make_preset_callback(self, idx):
        def _cb(_event):
            self.selected_preset = idx
            label, cfg = self.presets[idx]
            if cfg is not None:
                # Update custom sliders to match preset
                self.custom_sliders["p"].set_val(cfg.p)
                self.custom_sliders["e"].set_val(cfg.e)
                self.custom_sliders["n_orbits"].set_val(cfg.n_orbits)
                self.custom_sliders["n_output"].set_val(cfg.n_output)
            self._set_status(f"Selected preset: {label}")
        return _cb

    # Custom controls ---------------------------------------------------

    def _build_custom_controls(self):
        # Use axes created manually for better positioning
        # p slider
        ax_p = self.fig.add_axes([0.55, 0.60, 0.35, 0.03])
        s_p = Slider(ax_p, "p", 30.0, 200.0, valinit=100.0, valstep=5.0)
        self.custom_sliders["p"] = s_p

        # e slider
        ax_e = self.fig.add_axes([0.55, 0.55, 0.35, 0.03])
        s_e = Slider(ax_e, "e", 0.1, 0.9, valinit=0.4, valstep=0.02)
        self.custom_sliders["e"] = s_e

        # n_orbits slider (we'll cast to int)
        ax_no = self.fig.add_axes([0.55, 0.50, 0.35, 0.03])
        s_no = Slider(ax_no, "n_orbits", 2, 12, valinit=6, valstep=1)
        self.custom_sliders["n_orbits"] = s_no

        # n_output slider
        ax_nout = self.fig.add_axes([0.55, 0.45, 0.35, 0.03])
        s_nout = Slider(ax_nout, "n_output", 2000, 16000, valinit=8000, valstep=1000)
        self.custom_sliders["n_output"] = s_nout

        # JC couplings
        ax_sr = self.fig.add_axes([0.55, 0.35, 0.35, 0.03])
        s_sr = Slider(ax_sr, "s_ρ", 0.0, 2.0, valinit=1.0, valstep=0.1)
        self.custom_sliders["s_rho"] = s_sr

        ax_sp = self.fig.add_axes([0.55, 0.30, 0.35, 0.03])
        s_sp = Slider(ax_sp, "s_φ", 0.0, 2.0, valinit=0.6, valstep=0.1)
        self.custom_sliders["s_phi"] = s_sp

        ax_sv = self.fig.add_axes([0.55, 0.25, 0.35, 0.03])
        s_sv = Slider(ax_sv, "s_v", 0.0, 2.0, valinit=0.4, valstep=0.1)
        self.custom_sliders["s_v"] = s_sv

        # Text blurb
        self.ax_custom.text(
            0.0,
            0.65,
            "Adjust sliders to define a custom orbit.\n"
            "Preset selection will update these values.",
            transform=self.ax_custom.transAxes,
            fontsize=9,
            va="bottom",
        )

    # Run / status ------------------------------------------------------

    def _build_run_controls(self):
        ax_btn = self.fig.add_axes([0.1, 0.22, 0.20, 0.06])
        self.run_button = Button(ax_btn, "Run Simulation")
        self.run_button.on_clicked(self._on_run_clicked)

        self.status_text = self.ax_run.text(
            0.55,
            0.4,
            "Ready.",
            transform=self.ax_run.transAxes,
            fontsize=10,
            family="monospace",
            va="center",
        )

        self.ax_run.text(
            0.05,
            0.001,
            "Pipeline:\n"
            "  1) Load / solve complexity exponent\n"
            "  2) Run unified trajectory\n"
            "  3) Launch offline viewer",
            transform=self.ax_run.transAxes,
            fontsize=9,
            va="bottom",
        )

    def _set_status(self, msg: str):
        self.status_text.set_text(msg)
        self.fig.canvas.draw_idle()

    def _gather_run_config(self) -> RunConfig:
        """Collect current slider values into a RunConfig."""
        p = float(self.custom_sliders["p"].val)
        e = float(self.custom_sliders["e"].val)
        n_orbits = int(self.custom_sliders["n_orbits"].val)
        n_output = int(self.custom_sliders["n_output"].val)
        s_rho = float(self.custom_sliders["s_rho"].val)
        s_phi = float(self.custom_sliders["s_phi"].val)
        s_v = float(self.custom_sliders["s_v"].val)

        return RunConfig(
            p=p,
            e=e,
            n_orbits=n_orbits,
            n_output=n_output,
            s_rho=s_rho,
            s_phi=s_phi,
            s_v=s_v,
            with_jc=True,
            use_quantum_time=True,
        )

    def _on_run_clicked(self, _event):
        """Callback: start offline simulation, then open viewer."""
        self.run_button.label.set_text("Running...")
        self._set_status("Starting offline simulation...")
        self.fig.canvas.draw_idle()

        run_cfg = self._gather_run_config()

        def status_cb(msg: str):
            self._set_status(msg)

        results = run_offline_simulation(run_cfg, status_cb=status_cb)

        # Build viewer and show it
        self.viewer = ResultsViewer(results)
        self.viewer.show()

        # Now close the selection screen
        plt.close(self.fig)

    def run(self):
        plt.show()


# ======================================================================
#  RESULTS VIEWER
# ======================================================================

class ResultsViewer:
    """
    Offline replay / graph explorer.

    - Precomputes all static curves on init.
    - Maintains a time index i.
    - Animation step updates:
        * vertical time lines in all time-series plots
        * orbit marker
        * Bloch arrow & trail
    - Adding new graphs is plug-and-play: just add an axis and wire it
      into draw_static() and update_time_cursor().
    """

    def __init__(self, results: SimulationResults):
        self.res = results
        self.cfg = None  # not needed; use res.run_config & metrics

        # Time index state
        self.i = 0
        self.playing = True
        self.speed = 10  # indices per frame

        # Figure & axes
        self.fig = plt.figure(figsize=(22, 12))

        # Global style knobs
        TITLE_FONTSIZE = 9
        LABEL_FONTSIZE = 7
        TICK_FONTSIZE = 6
        MONO_FONTSIZE = 7

        self.MONO_FONTSIZE = MONO_FONTSIZE

        plt.rcParams.update({
            "font.size": LABEL_FONTSIZE,
            "axes.titlesize": TITLE_FONTSIZE,
            "axes.labelsize": LABEL_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
        })

        self.fig.suptitle(
            "QUANTUM GRAVITY LAB — GR Emergence from Quantum Clocks (Offline)",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        gs_main = gridspec.GridSpec(
            1,
            3,
            width_ratios=[1, 1.2, 1],
            left=0.05,
            right=0.95,
            top=0.94,
            bottom=0.16,
            wspace=0.25,
        )

        # LEFT: GR metrics
        gs_left = gridspec.GridSpecFromSubplotSpec(
            4, 1, subplot_spec=gs_main[0], hspace=0.35
        )
        self.ax_precession = self.fig.add_subplot(gs_left[0])
        self.ax_redshift = self.fig.add_subplot(gs_left[1])
        self.ax_energy = self.fig.add_subplot(gs_left[2])
        self.ax_L = self.fig.add_subplot(gs_left[3])

        # CENTER: orbit + complexity + clock error
        gs_center = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=gs_main[1], height_ratios=[2, 1, 1], hspace=0.3
        )
        self.ax_orbit = self.fig.add_subplot(gs_center[0])
        self.ax_complexity = self.fig.add_subplot(gs_center[1])
        self.ax_clock_err = self.fig.add_subplot(gs_center[2])

        # RIGHT: quantum system
        gs_right = gridspec.GridSpecFromSubplotSpec(
            3, 2, subplot_spec=gs_main[2],
            height_ratios=[1.5, 1, 1],
            hspace=0.35,
            wspace=0.3,
        )
        self.ax_bloch = self.fig.add_subplot(gs_right[0, 0], projection="3d")
        self.ax_photons = self.fig.add_subplot(gs_right[1, :])
        self.ax_coherence = self.fig.add_subplot(gs_right[2, :])
        self.ax_jc_params = self.fig.add_subplot(gs_right[0, 1])

        # Time slider & buttons
        self.slider_time = None
        self.slider_speed = None
        self.button_play = None

        # Artists that move with time
        self.time_cursor_lines = []  # vertical lines on time-series axes
        self.orbit_marker = None  # current point on orbit

        # Build UI
        self._setup_axes()
        self._draw_static()
        self._setup_controls()
        self._update_time_cursor(0)  # initial position
        self._draw_bottom_text()

        self.fig.subplots_adjust(
            top=0.93,
            bottom=0.08,
            left=0.04,
            right=0.98,
            hspace=0.45,
            wspace=0.3,
        )

        # Animation
        self.anim = FuncAnimation(
            self.fig,
            self._animate,
            frames=10_000,
            interval=50,
            blit=False,
            repeat=True,
        )

        self.trail_length = 150

    def _setup_axes(self):
        for ax in [
            self.ax_precession,
            self.ax_redshift,
            self.ax_energy,
            self.ax_L,
            self.ax_orbit,
            self.ax_complexity,
            self.ax_clock_err,
            self.ax_photons,
            self.ax_coherence,
            self.ax_jc_params,
        ]:
            ax.grid(True, alpha=0.3)

        # Bloch sphere wireframe
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)
        x_s = np.outer(np.cos(u), np.sin(v))
        y_s = np.outer(np.sin(u), np.sin(v))
        z_s = np.outer(np.ones_like(u), np.cos(v))
        self.ax_bloch.plot_wireframe(x_s, y_s, z_s, alpha=0.1, color="gray", linewidth=0.3)
        self.ax_bloch.set_xlim([-1.2, 1.2])
        self.ax_bloch.set_ylim([-1.2, 1.2])
        self.ax_bloch.set_zlim([-1.2, 1.2])
        self.ax_bloch.set_box_aspect([1, 1, 1])
        self.ax_bloch.set_title("Quantum State")

    def _draw_static(self):
        """Static curves (only time cursor and markers move)."""
        r = self.res
        t = r.t

        # LEFT: Precession
        if r.peri_angles.size > 0:
            n_peri = np.arange(r.peri_angles.size)
            self.ax_precession.plot(n_peri, r.peri_angles, "bo-", markersize=4, label="Measured")
            if r.precession_theory != 0.0:
                theory_step = r.precession_theory + 2.0 * math.pi
                theory_line = r.peri_angles[0] + theory_step * n_peri
                self.ax_precession.plot(n_peri, theory_line, "r--", label="GR theory", alpha=0.7)
                self.ax_precession.legend(fontsize=7)
        self.ax_precession.set_ylabel("Angle (rad)")
        self.ax_precession.set_title("Perihelion Precession")

        # LEFT: Time dilation
        self.ax_redshift.plot(t, r.clock_q, "b-", label="Quantum", linewidth=1)
        self.ax_redshift.plot(t, r.clock_gr, "r--", label="Schwarzschild", linewidth=1)
        self.ax_redshift.set_ylabel("dτ/dt")
        self.ax_redshift.set_title("Time Dilation")
        self.ax_redshift.legend(fontsize=7, loc="upper right")

        # LEFT: Energy
        self.ax_energy.plot(t, r.E, "g-", linewidth=1)
        self.ax_energy.axhline(r.E[0], color="gray", linestyle=":", alpha=0.5)
        self.ax_energy.set_ylabel("E")
        self.ax_energy.set_title("Orbital Energy")

        # LEFT: Angular momentum
        self.ax_L.plot(t, r.L, "purple", linewidth=1)
        self.ax_L.set_ylabel("L")
        self.ax_L.set_xlabel("Time")
        self.ax_L.set_title("Angular Momentum")

        # CENTER: Orbit
        sc = self.ax_orbit.scatter(
            r.x,
            r.y,
            c=r.C_Q,
            s=2,
            cmap="coolwarm",
            alpha=0.3,
        )
        self.fig.colorbar(sc, ax=self.ax_orbit, fraction=0.046, pad=0.04, label="C_Q")
        self.ax_orbit.plot(0, 0, "ko", markersize=8)

        # Peri/apo circles (pure Kepler)
        p = r.run_config.p
        e = r.run_config.e
        r_peri = p / (1.0 + e)
        r_apo = p / (1.0 - e)
        circle_peri = plt.Circle((0, 0), r_peri, fill=False, color="g", linestyle="--", alpha=0.9)
        circle_apo = plt.Circle((0, 0), r_apo, fill=False, color="r", linestyle="--", alpha=0.9)
        self.ax_orbit.add_patch(circle_peri)
        self.ax_orbit.add_patch(circle_apo)

        self.ax_orbit.set_aspect("equal")
        self.ax_orbit.set_xlabel("x [GM/c²]")
        self.ax_orbit.set_ylabel("y [GM/c²]")
        self.ax_orbit.set_title("Orbital Trajectory")

        # Orbit marker
        self.orbit_marker, = self.ax_orbit.plot([], [], "yo", markersize=6, markeredgecolor="k")

        # CENTER: Complexity
        self.ax_complexity.plot(t, r.C_Q, "orange", linewidth=1.5)
        self.ax_complexity.set_ylabel("C_Q")
        self.ax_complexity.set_title("Quantum Complexity")

        # CENTER: Clock error
        self.ax_clock_err.plot(t, r.clock_q - r.clock_gr, "purple", linewidth=1)
        self.ax_clock_err.axhline(0, color="black", linestyle="-", alpha=0.3)
        self.ax_clock_err.set_xlabel("Time")
        self.ax_clock_err.set_ylabel("Δ(dτ/dt)")
        self.ax_clock_err.set_title("Clock Error (Quantum - GR)")

        # RIGHT: photons
        self.ax_photons.plot(t, r.photons, linewidth=2)
        self.ax_photons.axhline(r.photons[0], color="gray", linestyle=":", alpha=0.5)
        self.ax_photons.set_ylabel("⟨n⟩")
        self.ax_photons.set_title("Photon Number")

        # RIGHT: coherence
        self.ax_coherence.plot(t, r.coherence, linewidth=1.5)
        self.ax_coherence.set_ylabel("|⟨σ⟩|")
        self.ax_coherence.set_title("Quantum Coherence")

        # RIGHT: JC params
        self.ax_jc_params.plot(t, r.omega_c, "b-", label="ω_cavity/ω₀", linewidth=1)
        self.ax_jc_params.plot(t, r.omega_a, "r-", label="ω_atom/ω₀", linewidth=1)
        self.ax_jc_params.set_xlabel("Time")
        self.ax_jc_params.set_ylabel("ω/ω₀")
        self.ax_jc_params.set_title("JC Parameters")
        self.ax_jc_params.legend(fontsize=7)

        # Time cursor lines on time-series axes
        self.time_cursor_lines = []
        for ax in [self.ax_redshift, self.ax_energy, self.ax_L,
                   self.ax_complexity, self.ax_clock_err,
                   self.ax_photons, self.ax_coherence, self.ax_jc_params]:
            line = ax.axvline(
                t[0] if len(t) > 0 else 0.0,
                color="0.3",
                linestyle="--",
                linewidth=0.5,
                alpha=0.7,
            )
            self.time_cursor_lines.append(line)

        # Bloch dynamic artists are handled in _update_bloch()

    def _draw_bottom_text(self):
        # left: accuracy block
        accuracy = getattr(self.res, "accuracy_scores", {}) or {}
        acc_lines = ["ACCURACY SCORES", "──────────────"]
        for key in ("precession", "clock", "energy", "overall"):
            if key in accuracy:
                acc_lines.append(f"{key:10s}: {accuracy[key]:5.1f}%")
        acc_text = "\n".join(acc_lines)

        self.fig.text(
            0.02, 0.03, acc_text,
            family="monospace",
            fontsize=self.MONO_FONTSIZE,
            va="bottom",
            ha="left"
        )

        # center: run config
        cfg = self.res.run_config
        cfg_lines = [
            "RUN CONFIG",
            "──────────",
            f"p          = {cfg.p:7.3f}",
            f"e          = {cfg.e:7.3f}",
            f"n_orbits   = {cfg.n_orbits:7d}",
            f"n_output   = {cfg.n_output:7d}",
            f"complexity = {self.res.complexity_power:7.4f}",
            f"alpha_gr   = {self.res.alpha_gr:7.3e}",
        ]
        if self.res.k_rq_ratio is not None:
            cfg_lines.append(f"k_rq_ratio = {self.res.k_rq_ratio:7.3f}")
        cfg_lines.append(f"r_q_trans. = {self.res.r_q_transition:7.3f}")
        cfg_text = "\n".join(cfg_lines)

        self.fig.text(
            0.36, 0.03, cfg_text,
            family="monospace",
            fontsize=self.MONO_FONTSIZE,
            va="bottom",
            ha="left"
        )

    def _setup_controls(self):
        # Time slider (0 .. N-1)
        N = len(self.res.t)
        ax_time = plt.axes([0.58, 0.02, 0.35, 0.015])
        self.slider_time = Slider(
            ax_time, "Time index", 0, max(0, N - 1),
            valinit=0,
            valstep=1,
        )
        self.slider_time.on_changed(self._on_time_slider)

        # Speed slider
        ax_speed = plt.axes([0.70, 0.06, 0.15, 0.02])
        self.slider_speed = Slider(
            ax_speed, "Speed", 1, 200,
            valinit=self.speed,
            valstep=1,
        )
        self.slider_speed.on_changed(self._on_speed_slider)

        # Play/pause button
        ax_play = plt.axes([0.09, 0.055, 0.08, 0.04])
        self.button_play = Button(ax_play, "Pause")
        self.button_play.on_clicked(self._toggle_play)

    def _on_time_slider(self, val):
        self.i = int(val)
        self._update_time_cursor(self.i)
        self.fig.canvas.draw_idle()

    def _on_speed_slider(self, val):
        self.speed = int(val)

    def _toggle_play(self, _event):
        self.playing = not self.playing
        self.button_play.label.set_text("Pause" if self.playing else "Play")

    def _update_time_cursor(self, i: int):
        r = self.res
        N = len(r.t)
        if N == 0:
            return
        i = max(0, min(i, N - 1))
        self.i = i

        t_i = r.t[i]

        # Time-series vertical lines
        for line in self.time_cursor_lines:
            line.set_xdata([t_i, t_i])

        # Orbit marker
        self.orbit_marker.set_data([r.x[i]], [r.y[i]])

        self._update_bloch(i)

    def _update_bloch(self, idx: int):
        """Update Bloch arrow and trail without disturbing static wireframe."""
        # Remove only dynamic artists (arrow + trail), keep wireframe
        artists_to_remove = []
        for art in list(self.ax_bloch.lines) + list(self.ax_bloch.collections):
            if getattr(art, "_is_dynamic", False):
                artists_to_remove.append(art)
        for art in artists_to_remove:
            art.remove()

        sx, sy, sz = self.res.bloch[idx]

        # Arrow
        q = self.ax_bloch.quiver(
            0, 0, 0, sx, sy, sz,
            arrow_length_ratio=0.1,
            linewidth=2,
            color="red",
        )
        q._is_dynamic = True

        # Short trail: only last `trail_length` points
        if idx > 0:
            start = max(0, idx - self.trail_length)
            trail = np.array(self.res.bloch[start: idx + 1])
            line, = self.ax_bloch.plot(
                trail[:, 0], trail[:, 1], trail[:, 2],
                "r-", linewidth=1, alpha=0.3,
            )
            line._is_dynamic = True

    def _animate(self, _frame):
        if not self.playing:
            return []

        if len(self.res.t) == 0:
            return []

        self.i = (self.i + self.speed) % len(self.res.t)
        self.slider_time.set_val(self.i)  # will call _update_time_cursor internally
        return []

    def show(self):
        # Just show this figure in the already-running Qt loop
        self.fig.show()


# ======================================================================
#  ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("QUANTUM GRAVITY LAB — OFFLINE EDITION")
    print("Run selection → calibration → unified trajectory → viewer")
    print("=" * 70)
    ui = RunSelectionUI()
    ui.run()

