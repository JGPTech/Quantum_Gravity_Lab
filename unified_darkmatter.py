#!/usr/bin/env python3
"""
Quantum Gravity Lab — DARK MATTER EDITION
=========================================

A high-contrast, dark-sector telemetry console wrapped around your
unified_sandbox_core pipeline:

  • Stage 1: LaunchConsole (mission profiles + parameter sliders)
  • Stage 2: run_offline_simulation(...) — full GR+QC+JC pipeline
  • Stage 3: TelemetryViewer — dark-matter-style dashboard

Assumes you have:
  - unified_sandbox_core.py  (Config, canonical_periapsis_ic, etc.)
  - calibrate.py             (calibrate_complexity_power)
in the same directory.
"""

import math
import os
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.linalg import expm

# --- Core imports from your framework ---------------------------------
from unified_sandbox_core import (
    Config,
    canonical_periapsis_ic,
    run_unified_trajectory,
    compute_quantum_clock_rate,
)
from calibrate import calibrate_complexity_power


# ======================================================================
#  THEME CONFIGURATION — DARK MATTER HUD
# ======================================================================
ANIM_REFS = []  # keep strong refs to animations so Matplotlib doesn't whine

THEME = {
    "bg": "#05060a",         # Deep void
    "panel": "#11151c",      # HUD panel
    "fg": "#e5e9f0",         # Primary text
    "cyan": "#66fcf1",       # Main telemetry
    "cyan_dim": "#45a29e",
    "magenta": "#ff00ff",    # Theory / alerts
    "yellow": "#f2a900",     # Complexity / dark sector
    "green": "#0aff0a",      # Good
    "red": "#ff4b4b",        # Bad
    "grid": "#2d3436",
    "font": "monospace",
}

plt.rcParams.update({
    "toolbar": "None",
    "font.family": THEME["font"],
    "text.color": THEME["fg"],
    "axes.facecolor": THEME["bg"],
    "axes.edgecolor": THEME["cyan_dim"],
    "axes.labelcolor": THEME["fg"],
    "axes.titlecolor": THEME["cyan"],
    "xtick.color": THEME["cyan_dim"],
    "ytick.color": THEME["cyan_dim"],
    "grid.color": THEME["grid"],
    "figure.facecolor": THEME["bg"],
    "legend.frameon": False,
    "savefig.facecolor": THEME["bg"],
    "savefig.edgecolor": THEME["bg"],
})


# ======================================================================
#  DATA STRUCTURES
# ======================================================================
CALIBRATION_CSV = "complexity_power_grid_scan.csv"


@dataclass
class RunConfig:
    """Parameters selected in LaunchConsole."""
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
    """
    Offline run payload consumed by TelemetryViewer.
    Mirrors the working 'offline edition' structure.
    """
    # core config / coupling
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

    # JC / quantum telemetry
    bloch: np.ndarray           # (N, 3)
    photons: np.ndarray         # (N,)
    coherence: np.ndarray       # (N,)
    omega_c: np.ndarray         # (N,)
    omega_a: np.ndarray         # (N,)


# ======================================================================
#  CALIBRATION HELPERS  (upgraded: keyed by full slider config)
# ======================================================================
CALIBRATION_CSV = "complexity_power_grid_scan.csv"


def load_calibration_df() -> pd.DataFrame:
    """
    Load the calibration table, upgrading older CSVs by adding any
    missing columns with sensible defaults.
    """
    if os.path.exists(CALIBRATION_CSV):
        df = pd.read_csv(CALIBRATION_CSV)
    else:
        df = pd.DataFrame()

    # Full schema we now care about
    cols_defaults = {
        "p": np.nan,
        "e": np.nan,
        "n_orbits": np.nan,
        "n_output": np.nan,
        "s_rho": np.nan,
        "s_phi": np.nan,
        "s_v": np.nan,
        "use_quantum_time": False,
        "with_jc": True,
        "ok": True,
        "complexity_power": np.nan,
        "rel_error": np.nan,
        "measured_shift": np.nan,
        "theory_shift": np.nan,
        "n_peri": 0,
        "k_rq_ratio": np.nan,
        "r_q_transition": np.nan,
    }

    # Add any missing columns
    for col, default in cols_defaults.items():
        if col not in df.columns:
            df[col] = default

    return df


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
    use_quantum_time: bool,
    with_jc: bool,
) -> pd.Series | None:
    """
    Look up an existing calibration row that matches the full slider
    configuration. If any slider (or orbit sampling) changes, we treat
    it as a distinct calibration point.
    """
    if df.empty:
        return None

    mask = (
        np.isclose(df["p"], p)
        & np.isclose(df["e"], e)
        & np.isclose(df["n_orbits"], n_orbits)
        & np.isclose(df["n_output"], n_output)
        & np.isclose(df["s_rho"], s_rho)
        & np.isclose(df["s_phi"], s_phi)
        & np.isclose(df["s_v"], s_v)
        & (df["use_quantum_time"] == bool(use_quantum_time))
        & (df["with_jc"] == bool(with_jc))
    )

    rows = df[mask]
    if rows.empty:
        return None

    # If multiple rows exist (old junk etc.), pick the one with smallest |rel_error|
    if "rel_error" in rows.columns:
        idx = rows["rel_error"].abs().idxmin()
    else:
        idx = rows.index[0]

    return rows.loc[idx]


def get_or_calibrate_complexity_power(
    p: float,
    e: float,
    n_orbits: int,
    n_output: int,
    s_rho: float,
    s_phi: float,
    s_v: float,
    use_quantum_time: bool,
    with_jc: bool,
    df_calib: pd.DataFrame,
) -> tuple[float, pd.DataFrame, dict]:
    """
    Look up complexity_power for the full (p, e, slider config) combo.
    If missing, call calibrate_complexity_power *for that combo* and
    append to the CSV.

    This means:
      - change ANY slider (p, e, n_orbits, n_output, s_rho, s_phi, s_v,
        use_quantum_time, with_jc)
      - and we will retrain + store a fresh row.
    """
    row = find_calibrated_row(
        df_calib,
        p=p,
        e=e,
        n_orbits=n_orbits,
        n_output=n_output,
        s_rho=s_rho,
        s_phi=s_phi,
        s_v=s_v,
        use_quantum_time=use_quantum_time,
        with_jc=with_jc,
    )

    if row is not None and bool(row.get("ok", True)):
        cp = float(row["complexity_power"])
        return cp, df_calib, row.to_dict()

    # Need to calibrate for this exact configuration
    print(
        f"[CAL] No record for "
        f"(p={p:.3f}, e={e:.3f}, n_orbits={n_orbits}, n_output={n_output}, "
        f"s_rho={s_rho:.3f}, s_phi={s_phi:.3f}, s_v={s_v:.3f}, "
        f"use_qt={use_quantum_time}, with_jc={with_jc}) "
        f"→ running calibrate_complexity_power"
    )

    # NOTE: calibrate_complexity_power itself only cares about (p,e,n_orbits,n_output),
    # but we still key + store with the full slider config.
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

    record = {
        "p": float(res.get("p", p)),
        "e": float(res.get("e", e)),
        "n_orbits": int(res.get("n_orbits", n_orbits)),
        "n_output": int(res.get("n_output", n_output)),
        "s_rho": float(s_rho),
        "s_phi": float(s_phi),
        "s_v": float(s_v),
        "use_quantum_time": bool(use_quantum_time),
        "with_jc": bool(with_jc),
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
#  JC HELPERS — same as offline edition, reused
# ======================================================================

def initialize_jc_state(dim: int, n_cavity: int) -> np.ndarray:
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

    norm = math.sqrt(sigma_x**2 + sigma_y**2 + sigma_z**2)
    if norm > 0:
        return sigma_x / norm, sigma_y / norm, sigma_z / norm
    else:
        return 0.0, 0.0, 1.0


def jc_expectations(rho: np.ndarray, n_cavity: int) -> tuple[float, float]:
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
#  GR METRICS & CLOCKS (same as offline edition)
# ======================================================================

def schwarzschild_clock(r: float, G: float, M: float, c: float) -> float:
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
    r = np.sqrt(x**2 + y**2 + z**2)
    v2 = vx**2 + vy**2 + vz**2

    is_min = (r[1:-1] < r[:-2]) & (r[1:-1] < r[2:])
    min_indices = np.where(is_min)[0] + 1

    gr_metrics: dict = {}
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

    E = 0.5 * v2 - cfg.G * cfg.M / np.maximum(r, 1e-8)
    if np.mean(E) != 0.0:
        energy_variation = float(np.std(E) / abs(np.mean(E)))
    else:
        energy_variation = float(np.std(E))
    gr_metrics["energy_variation"] = energy_variation

    if len(clock_q) == len(clock_gr) and len(clock_q) > 0:
        gr_metrics["max_dilation"] = 1.0 - float(np.min(clock_q))
        gr_metrics["max_dilation_gr"] = 1.0 - float(np.min(clock_gr))

    gr_metrics["peri_angles"] = peri_angles

    scores: dict[str, float] = {}

    # Precession score
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

    # Clock score
    if len(clock_q) == len(clock_gr) and len(clock_q) > 0:
        clock_error = float(np.mean(np.abs(clock_q - clock_gr)))
        scores["clock"] = max(0.0, 100.0 * (1.0 - clock_error))
    else:
        scores["clock"] = 0.0

    # Energy conservation score
    ev = gr_metrics.get("energy_variation", 1.0)
    scores["energy"] = max(0.0, 100.0 * (1.0 - ev))

    scores["overall"] = float(np.mean(list(scores.values()))) if scores else 0.0

    return gr_metrics, scores


# ======================================================================
#  OFFLINE SIMULATION RUNNER — full framework + dark-sector narrative
# ======================================================================

def run_offline_simulation(run_cfg: RunConfig, status_cb=None) -> SimulationResults:
    """
    Full offline pipeline, identical in physics to the retro UI:
      - load / update calibration grid
      - solve complexity_power (dark sector coupling exponent)
      - run unified trajectory
      - evolve JC state along path
      - compute GR metrics + accuracy
    """

    def log(msg: str):
        if status_cb is not None:
            status_cb(msg)
        print(msg)

    # 1) Calibration table
    log("> LOADING CALIBRATION MANIFOLD …")
    df_calib = load_calibration_df()

    # 2) Solve / load complexity exponent (keyed by full slider config)
    log(
        f"> SOLVING DARK COMPLEXITY for "
        f"p={run_cfg.p:.3f}, e={run_cfg.e:.3f}, "
        f"n_orbits={run_cfg.n_orbits}, n_output={run_cfg.n_output}, "
        f"s_rho={run_cfg.s_rho:.2f}, s_phi={run_cfg.s_phi:.2f}, s_v={run_cfg.s_v:.2f}"
    )

    cp, df_calib, calib_record = get_or_calibrate_complexity_power(
        p=run_cfg.p,
        e=run_cfg.e,
        n_orbits=run_cfg.n_orbits,
        n_output=run_cfg.n_output,
        s_rho=run_cfg.s_rho,
        s_phi=run_cfg.s_phi,
        s_v=run_cfg.s_v,
        use_quantum_time=run_cfg.use_quantum_time,
        with_jc=run_cfg.with_jc,
        df_calib=df_calib,
    )


    # 3) Configure orbit
    log("> CONFIGURING UNIFIED SANDBOX …")
    cfg = Config()
    cfg.p = float(run_cfg.p)
    cfg.e = float(run_cfg.e)
    cfg.complexity_power = float(cp)

    # Dynamic k_rq_ratio if enabled
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

    # Time span
    a = cfg.p / (1.0 - cfg.e**2)
    T_orb = 2.0 * math.pi * math.sqrt(a**3 / (cfg.G * cfg.M))
    t_max = run_cfg.n_orbits * T_orb

    # Initial conditions
    r0, v0 = canonical_periapsis_ic(cfg)

    # 4) Unified trajectory
    log("> INTEGRATING ORBIT (GR+Dark+QC) …")
    df = run_unified_trajectory(
        cfg,
        r0,
        v0,
        t_max,
        n_output=run_cfg.n_output,
        with_jc=run_cfg.with_jc,
        use_quantum_time=run_cfg.use_quantum_time,
    )

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
    log("> COMPUTING DARK COMPLEXITY & CLOCKS …")
    C_Q_list: list[float] = []
    clock_q_list: list[float] = []
    clock_gr_list: list[float] = []

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

    # 5) JC evolution
    log("> EVOLVING QUBIT–CAVITY SYSTEM …")
    omega_cavity_0 = 2.0 * math.pi * 1.0
    omega_atom_0 = 2.0 * math.pi * 1.0
    g_0 = 0.09 * 2.0 * math.pi
    n_cavity = 20

    s_rho = run_cfg.s_rho
    s_phi = run_cfg.s_phi
    s_v = run_cfg.s_v

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

    # 6) GR metrics + accuracy
    log("> EVALUATING GR TELEMETRY & DARK BUDGET …")
    gr_metrics, accuracy_scores = compute_gr_metrics_and_accuracy(
        cfg, t, x, y, z, vx, vy, vz, C_Q, clock_q, clock_gr
    )

    peri_angles = gr_metrics.get("peri_angles", np.array([]))
    precession_measured = float(gr_metrics.get("precession_measured", 0.0))
    precession_theory = float(gr_metrics.get("precession_theory", 0.0))
    energy_variation = float(gr_metrics.get("energy_variation", 0.0))
    clock_max_dilation = float(gr_metrics.get("max_dilation", 0.0))
    clock_max_dilation_gr = float(gr_metrics.get("max_dilation_gr", 0.0))

    log("> OFFLINE RUN COMPLETE. HANDING OFF TO TELEMETRY VIEWER …")
    time.sleep(0.3)

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
#  PART 1: LAUNCH CONSOLE  (Dark-matter mission select)
# ======================================================================

class LaunchConsole:
    """
    Dark-matter-flavored loading screen:
      - left: mission profiles
      - right: orbital sliders
      - bottom: scrolling system log
    """

    def __init__(self):
        self.fig = plt.figure(figsize=(10, 6))
        try:
            self.fig.canvas.manager.set_window_title("QGL :: DARK MATTER LAUNCH CONSOLE")
        except Exception:
            pass

        gs = gridspec.GridSpec(
            3, 2,
            width_ratios=[1, 1.5],
            height_ratios=[1, 4, 2],
            left=0.06,
            right=0.96,
            top=0.92,
            bottom=0.10,
            wspace=0.25,
            hspace=0.25,
        )

        # Header
        ax_header = self.fig.add_subplot(gs[0, :])
        ax_header.axis("off")
        ax_header.text(
            0.01, 0.65,
            "QUANTUM GRAVITY LAB — DARK MATTER EDITION",
            fontsize=18,
            color=THEME["cyan"],
            fontweight="bold",
        )
        ax_header.text(
            0.01, 0.25,
            "// System ready – select a mission profile or define custom orbit.",
            fontsize=9,
            color=THEME["fg"],
        )

        # Mission presets
        self.ax_presets = self.fig.add_subplot(gs[1, 0])
        self.ax_presets.axis("off")
        self.ax_presets.set_title(
            "MISSION PROFILES",
            fontsize=10,
            color=THEME["magenta"],
            loc="left",
        )

        self.presets = [
            ("MERCURY PROTOCOL", RunConfig(p=80.0, e=0.206, n_orbits=8, n_output=8000)),
            ("HIGH ECCENTRICITY", RunConfig(p=100.0, e=0.4, n_orbits=6, n_output=6000)),
            ("EVENT HORIZON BRINK", RunConfig(p=60.0, e=0.8, n_orbits=6, n_output=8000)),
        ]

        self.preset_buttons: list[Button] = []
        for i, (label, _) in enumerate(self.presets):
            ax_btn = self.fig.add_axes([0.09, 0.60 - i * 0.09, 0.32, 0.06], facecolor=THEME["panel"])
            btn = Button(
                ax_btn,
                label,
                color=THEME["panel"],
                hovercolor=THEME["cyan_dim"],
            )
            btn.label.set_color(THEME["cyan"])
            btn.label.set_fontsize(9)
            btn.on_clicked(self._make_preset_callback(i))
            self.preset_buttons.append(btn)

        # Slider panel
        self.ax_sliders = self.fig.add_subplot(gs[1, 1])
        self.ax_sliders.axis("off")
        self.ax_sliders.set_title(
            "ORBITAL PARAMETERS",
            fontsize=10,
            color=THEME["magenta"],
            loc="left",
        )

        self.sliders: dict[str, Slider] = {}
        self._build_sliders()

        # Log panel
        self.ax_log = self.fig.add_subplot(gs[2, :])
        self.ax_log.set_facecolor(THEME["panel"])
        self.ax_log.set_xticks([])
        self.ax_log.set_yticks([])
        for spine in self.ax_log.spines.values():
            spine.set_visible(False)

        self.log_lines = [
            "> QGL DarkMatter subsystem online.",
            "> Awaiting mission selection …",
        ]
        self.log_text = self.ax_log.text(
            0.01,
            0.95,
            "",
            transform=self.ax_log.transAxes,
            color=THEME["cyan"],
            fontsize=8,
            va="top",
            ha="left",
            family="monospace",
        )
        self._update_log()

        # ENGAGE button
        ax_go = self.fig.add_axes([0.70, 0.04, 0.23, 0.07], facecolor=THEME["panel"])
        self.btn_run = Button(
            ax_go,
            "ENGAGE DARK SECTOR",
            color=THEME["cyan_dim"],
            hovercolor=THEME["cyan"],
        )
        self.btn_run.label.set_color(THEME["bg"])
        self.btn_run.label.set_weight("bold")
        self.btn_run.on_clicked(self._on_engage_clicked)

    # ---- helpers ------------------------------------------------------

    def _build_sliders(self):
        # (label, min, max, init, y-position)
        slider_specs = [
            ("p", 30.0, 200.0, 100.0, 0.70),
            ("e", 0.1, 0.9, 0.4, 0.63),
            ("n_orbits", 2, 12, 6, 0.56),
            ("n_output", 2000, 16000, 8000, 0.49),
            ("s_rho", 0.0, 2.0, 1.0, 0.40),
            ("s_phi", 0.0, 2.0, 0.6, 0.33),
            ("s_v", 0.0, 2.0, 0.4, 0.26),
        ]

        for name, vmin, vmax, vinit, ypos in slider_specs:
            ax_s = self.fig.add_axes(
                [0.56, ypos, 0.3, 0.035],
                facecolor=THEME["panel"],
            )
            slider = Slider(
                ax_s,
                f"{name} ",
                vmin,
                vmax,
                valinit=vinit,
                color=THEME["cyan"],
            )
            slider.label.set_color(THEME["fg"])
            slider.valtext.set_color(THEME["cyan"])
            self.sliders[name] = slider

    def _make_preset_callback(self, idx: int):
        def _cb(_event):
            label, cfg = self.presets[idx]
            self.sliders["p"].set_val(cfg.p)
            self.sliders["e"].set_val(cfg.e)
            self.sliders["n_orbits"].set_val(cfg.n_orbits)
            self.sliders["n_output"].set_val(cfg.n_output)
            self._log(f"> Mission profile loaded: {label}")
        return _cb

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_lines.append(f"[{ts}] {msg}")
        if len(self.log_lines) > 7:
            self.log_lines = self.log_lines[-7:]
        self._update_log()

    def _update_log(self):
        self.log_text.set_text("\n".join(self.log_lines))
        self.fig.canvas.draw_idle()

    def _gather_run_config(self) -> RunConfig:
        return RunConfig(
            p=float(self.sliders["p"].val),
            e=float(self.sliders["e"].val),
            n_orbits=int(self.sliders["n_orbits"].val),
            n_output=int(self.sliders["n_output"].val),
            s_rho=float(self.sliders["s_rho"].val),
            s_phi=float(self.sliders["s_phi"].val),
            s_v=float(self.sliders["s_v"].val),
            with_jc=True,
            use_quantum_time=True,
        )

    def _on_engage_clicked(self, _event):
        self.btn_run.label.set_text("RUNNING …")
        self.btn_run.color = THEME["panel"]
        self._log("> Dark-sector integration started")
        self.fig.canvas.draw()
        plt.pause(0.05)

        run_cfg = self._gather_run_config()

        def status_cb(msg: str):
            # Map pipeline messages into the log
            self._log(msg)

        results = run_offline_simulation(run_cfg, status_cb=status_cb)

        plt.close(self.fig)
        viewer = TelemetryViewer(results)
        viewer.show()

    def run(self):
        plt.show()


# ======================================================================
#  PART 2: TELEMETRY VIEWER  (Dark-matter dashboard)
# ======================================================================

class TelemetryViewer:
    """
    Dark-matter telemetry dashboard:
      - left: precession, time dilation, clock error
      - center: orbit colored by C_Q + complexity
      - right: Bloch sphere + photon / coherence traces
      - bottom: overlay with config + accuracy
    """

    def __init__(self, results: SimulationResults):
        self.res = results
        self.i = 0
        self.speed = 20
        self.playing = True

        self.fig = plt.figure(figsize=(20, 10))
        try:
            self.fig.canvas.manager.set_window_title("QGL :: DARK MATTER TELEMETRY")
        except Exception:
            pass

        gs = gridspec.GridSpec(
            3,
            3,
            width_ratios=[1, 1.6, 1.0],
            height_ratios=[1, 1, 1],
            left=0.04,
            right=0.98,
            top=0.93,
            bottom=0.10,
            wspace=0.25,
            hspace=0.35,
        )

        # Left column
        self.ax_prec = self._make_panel(gs[0, 0], "PERIHELION PRECESSION")
        self.ax_dil = self._make_panel(gs[1, 0], "TIME DILATION  dτ/dt")
        self.ax_err = self._make_panel(gs[2, 0], "CLOCK SYNC ERROR (Quantum − GR)")

        # Center column
        self.ax_orbit = self.fig.add_subplot(gs[0:2, 1])
        self.ax_orbit.set_aspect("equal")
        self.ax_orbit.set_title(
            "SPACETIME TRAJECTORY (colored by dark complexity C_Q)",
            fontsize=9,
            color=THEME["cyan"],
            loc="left",
        )
        self.ax_comp = self._make_panel(gs[2, 1], "DARK COMPLEXITY FIELD  C_Q(t)")

        # Right column
        self.ax_bloch = self.fig.add_subplot(gs[0, 2], projection="3d")
        self.ax_bloch.set_facecolor(THEME["bg"])
        self.ax_bloch.grid(False)
        self.ax_bloch.set_xticks([])
        self.ax_bloch.set_yticks([])
        self.ax_bloch.set_zticks([])
        self.ax_bloch.set_title("QUBIT STATE on Bloch Sphere", color=THEME["magenta"], fontsize=10)

        self.ax_phot = self._make_panel(gs[1, 2], "CAVITY PHOTON NUMBER  ⟨n⟩")
        self.ax_coh = self._make_panel(gs[2, 2], "COHERENCE MAGNITUDE  |σ|")

        # Time slider
        N = len(self.res.t)
        ax_slider = self.fig.add_axes([0.20, 0.04, 0.60, 0.02], facecolor=THEME["panel"])
        self.slider = Slider(
            ax_slider,
            "",
            0,
            max(0, N - 1),
            valinit=0,
            valstep=1,
            color=THEME["cyan"],
        )
        self.slider.valtext.set_visible(False)
        self.slider.on_changed(self._on_seek)

        # Play / pause small button
        ax_play = self.fig.add_axes([0.06, 0.035, 0.08, 0.04], facecolor=THEME["panel"])
        self.btn_play = Button(
            ax_play,
            "Pause",
            color=THEME["cyan_dim"],
            hovercolor=THEME["cyan"],
        )
        self.btn_play.label.set_color(THEME["bg"])
        self.btn_play.on_clicked(self._toggle_play)

        # Speed slider
        ax_speed = self.fig.add_axes([0.85, 0.04, 0.12, 0.02], facecolor=THEME["panel"])
        self.speed_slider = Slider(
            ax_speed,
            "speed",
            1,
            200,
            valinit=self.speed,
            valstep=1,
            color=THEME["cyan"],
        )
        self.speed_slider.label.set_color(THEME["fg"])
        self.speed_slider.valtext.set_color(THEME["cyan"])
        self.speed_slider.on_changed(self._on_speed_change)

        # Static plots & dynamic artists
        self.cursors = []
        self.orbit_head = None
        self.bloch_quiver = None
        self._render_static()
        self._setup_bloch()
        self._draw_overlays()

        # Bloch trail config
        self.trail_length = 150      # number of time steps to show in the tail
        self.bloch_quiver = None
        self.bloch_trail = None

        # Animation
        self.anim = FuncAnimation(
            self.fig,
            self._animate,
            frames=10_000,
            interval=40,
            blit=False,
        )
        ANIM_REFS.append(self.anim)



    # ---- helpers ------------------------------------------------------

    def _make_panel(self, grid_slice, title: str):
        ax = self.fig.add_subplot(grid_slice)
        ax.set_title(title, loc="left", fontsize=9, color=THEME["cyan_dim"])
        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.grid(True, alpha=0.25)
        return ax

    def _glow_plot(self, ax, x, y, color, lw=1.5):
        ax.plot(x, y, color=color, linewidth=lw * 3, alpha=0.25)
        ax.plot(x, y, color=color, linewidth=lw, alpha=1.0)

    def _render_static(self):
        r = self.res
        t = r.t

        # Precession
        if r.peri_angles.size > 0:
            n_peri = np.arange(r.peri_angles.size)
            self._glow_plot(self.ax_prec, n_peri, r.peri_angles, THEME["cyan"])
            if r.precession_theory != 0.0:
                theory_step = r.precession_theory + 2.0 * math.pi
                theory_line = r.peri_angles[0] + theory_step * n_peri
                self.ax_prec.plot(
                    n_peri,
                    theory_line,
                    linestyle="--",
                    color=THEME["magenta"],
                    alpha=0.6,
                    label="GR",
                )
                self.ax_prec.legend(fontsize=7, loc="upper left")

        # Time dilation
        self.ax_dil.plot(t, r.clock_gr, "--", color=THEME["magenta"], alpha=0.6, label="GR")
        self._glow_plot(self.ax_dil, t, r.clock_q, THEME["cyan"])
        self.ax_dil.set_ylabel("dτ/dt")
        self.ax_dil.legend(fontsize=7, loc="upper right")

        # Clock error
        err = r.clock_q - r.clock_gr
        color_err = THEME["green"] if np.max(np.abs(err)) < 1e-4 else THEME["red"]
        self.ax_err.plot(t, err, color=color_err, linewidth=1)
        self.ax_err.axhline(0.0, color=THEME["grid"], linestyle=":", linewidth=0.8)
        self.ax_err.set_ylim(-1e-4, 1e-4)
        self.ax_err.set_ylabel("Δ(dτ/dt)")

        # Orbit
        sc = self.ax_orbit.scatter(
            r.x,
            r.y,
            c=r.C_Q,
            s=3,
            cmap="inferno",
            alpha=0.85,
        )
        self.ax_orbit.plot(0, 0, marker="*", color=THEME["yellow"], markersize=9)
        cb = self.fig.colorbar(sc, ax=self.ax_orbit, fraction=0.046, pad=0.02)
        cb.set_label("dark complexity  C_Q", fontsize=7)

        # Kepler reference circles
        p = r.run_config.p
        e = r.run_config.e
        r_peri = p / (1.0 + e)
        r_apo = p / (1.0 - e)
        circle_peri = plt.Circle((0, 0), r_peri, fill=False, color=THEME["green"], linestyle="--", alpha=0.2)
        circle_apo = plt.Circle((0, 0), r_apo, fill=False, color=THEME["red"], linestyle="--", alpha=0.2)
        self.ax_orbit.add_patch(circle_peri)
        self.ax_orbit.add_patch(circle_apo)
        self.ax_orbit.set_xlabel("x [GM/c²]")
        self.ax_orbit.set_ylabel("y [GM/c²]")

        # Orbit head
        self.orbit_head, = self.ax_orbit.plot([], [], "o", color=THEME["cyan"], markersize=6)

        # Complexity
        self._glow_plot(self.ax_comp, t, r.C_Q, THEME["yellow"])
        self.ax_comp.set_ylabel("C_Q")

        # Photons & coherence
        self._glow_plot(self.ax_phot, t, r.photons, THEME["cyan"])
        self.ax_phot.set_ylabel("⟨n⟩")

        self._glow_plot(self.ax_coh, t, r.coherence, THEME["magenta"])
        self.ax_coh.set_ylabel("|σ|")

        # Time cursors on time-series axes
        self.cursors = []
        for ax in [self.ax_dil, self.ax_err, self.ax_comp, self.ax_phot, self.ax_coh]:
            line = ax.axvline(
                t[0] if len(t) > 0 else 0.0,
                color=THEME["fg"],
                linestyle=":",
                linewidth=0.7,
                alpha=0.6,
            )
            self.cursors.append(line)

    def _setup_bloch(self):
        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 16)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))
        self.ax_bloch.plot_wireframe(
            xs, ys, zs,
            color=THEME["cyan_dim"],
            linewidth=0.5,
            alpha=0.2,
        )
        self.ax_bloch.set_box_aspect([1, 1, 1])
        self.bloch_quiver = None

    def _draw_overlays(self):
        acc = self.res.accuracy_scores or {}
        cfg = self.res.run_config

    # ---- interaction / animation --------------------------------------

    def _animate(self, _frame):
        if not self.playing or len(self.res.t) == 0:
            return []
        self.i = (self.i + self.speed) % len(self.res.t)
        # This will call _on_seek via slider callback
        self.slider.set_val(self.i)
        return []

    def _on_seek(self, val):
        i = int(val)
        self.i = i
        r = self.res
        if len(r.t) == 0:
            return

        t_i = r.t[i]

        # Move time cursors
        for line in self.cursors:
            line.set_xdata([t_i, t_i])

        # Move orbit head
        self.orbit_head.set_data([r.x[i]], [r.y[i]])

        # Update Bloch arrow + trail
        if self.bloch_quiver is not None:
            self.bloch_quiver.remove()
        if getattr(self, "bloch_trail", None) is not None:
            self.bloch_trail.remove()

        sx, sy, sz = r.bloch[i]

        # Arrow (current state)
        self.bloch_quiver = self.ax_bloch.quiver(
            0, 0, 0,
            sx, sy, sz,
            color=THEME["red"],
            linewidth=2,
            arrow_length_ratio=0.15,
        )

        # Trail: last `trail_length` points on Bloch sphere
        if i > 0:
            start = max(0, i - self.trail_length)
            trail = r.bloch[start : i + 1]
            self.bloch_trail, = self.ax_bloch.plot(
                trail[:, 0],
                trail[:, 1],
                trail[:, 2],
                color=THEME["red"],
                linewidth=1.0,
                alpha=0.35,
            )


        self.fig.canvas.draw_idle()

    def _toggle_play(self, _event):
        self.playing = not self.playing
        self.btn_play.label.set_text("Pause" if self.playing else "Play")

    def _on_speed_change(self, val):
        self.speed = int(val)

    def show(self):
        plt.show()


# ======================================================================
#  ENTRY
# ======================================================================

if __name__ == "__main__":
    ui = LaunchConsole()
    ui.run()
