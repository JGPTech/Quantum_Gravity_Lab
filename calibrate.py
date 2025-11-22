#!/usr/bin/env python3
"""
calibrate_complexity_power.py

Offline calibration of the spatial complexity exponent `complexity_power`
for a given (p, e) orbit.

Goal:
  Find complexity_power such that the measured perihelion precession
  from the unified clock gravity simulation matches the GR 1PN prediction
  as closely as possible.

This does NOT touch the core module. It just imports your existing
UnifiedSandbox and runs a 1D search over complexity_power.
"""

import time
import numpy as np
from itertools import product
import pandas as pd

from unified_sandbox_core import (
    Config,
    canonical_periapsis_ic,
    run_unified_trajectory,
)


# ======================================================================
#  Helper: ensure cfg matches the target orbit
# ======================================================================

def configure_orbit(cfg: Config, p: float, e: float) -> None:
    """
    Update cfg to use the desired (p, e) and recompute derived parameters.

    This keeps:
      - k_rq_ratio in 'dynamic' mode (if enabled)
      - alpha_gr consistent with p
      - r_q_transition consistent with (p, e, k_rq)
    """
    cfg.p = float(p)
    cfg.e = float(e)

    # If using dynamic k_rq mode, refresh it for this (p, e)
    if getattr(cfg, "k_rq_ratio_mode", "fixed") == "dynamic":
        from unified_sandbox_core import k_rq_from_pe
        cfg.k_rq_ratio = k_rq_from_pe(cfg.p, cfg.e, default=cfg.k_rq_ratio_default)

    # Update GR coupling and transition radius
    L_sq = cfg.G * cfg.M * cfg.p
    cfg.alpha_gr = (3.0 * cfg.G * cfg.M * L_sq) / (cfg.c ** 2)
    cfg.update_r_q_transition_from_orbit()


# ======================================================================
#  Precession measurement for a given complexity_power
# ======================================================================

def measure_precession_error(
    cfg: Config,
    complexity_power: float,
    n_orbits: int = 8,
    n_output: int = 12000,
) -> dict:
    """
    Set cfg.complexity_power, run an orbit, and measure precession error.

    Returns a dict with:
      - 'complexity_power'
      - 'rel_error'       (relative error vs 1PN GR)
      - 'measured_shift'  (rad/orbit)
      - 'theory_shift'    (rad/orbit)
      - 'n_peri'          (# of detected periapsis passages)
      - 'ok'              (bool, True if measurement is usable)
    """
    cfg.complexity_power = float(complexity_power)

    # Initial conditions at periapsis for this cfg
    r0, v0 = canonical_periapsis_ic(cfg)

    # Orbital period (Keplerian) with current p, e, G
    a = cfg.p / (1.0 - cfg.e ** 2)
    T_orb = 2.0 * np.pi * np.sqrt(a ** 3 / cfg.G)
    t_max = n_orbits * T_orb

    # Run unified trajectory; GR strength comes from cfg.alpha_gr * C_Q(r)
    # We can disable JC here for speed; it doesn't affect the GR force.
    df = run_unified_trajectory(
        cfg,
        r0,
        v0,
        t_max,
        n_output=n_output,
        with_jc=True,
        use_quantum_time=True,
    )

    r = df["r"].values
    if r.size < 3:
        return {
            "complexity_power": complexity_power,
            "rel_error": np.inf,
            "measured_shift": 0.0,
            "theory_shift": 0.0,
            "n_peri": 0,
            "ok": False,
        }

    # Detect periapsis passages as local minima in r(t)
    is_min = (r[1:-1] < r[:-2]) & (r[1:-1] < r[2:])
    min_indices = np.where(is_min)[0] + 1

    if min_indices.size < 2:
        # Not enough periapsis points to estimate precession
        return {
            "complexity_power": complexity_power,
            "rel_error": np.inf,
            "measured_shift": 0.0,
            "theory_shift": 0.0,
            "n_peri": int(min_indices.size),
            "ok": False,
        }

    x = df["x"].values
    y = df["y"].values
    angles = np.unwrap(np.arctan2(y, x))
    peri_angles = angles[min_indices]

    diffs = np.diff(peri_angles)
    avg_shift_total = float(np.mean(diffs))
    avg_shift_excess = avg_shift_total - 2.0 * np.pi  # subtract pure 2π

    # 1PN GR prediction (same as in your validation script)
    theory_shift = (6.0 * np.pi * cfg.G * cfg.M) / (cfg.c ** 2 * cfg.p)
    if theory_shift == 0.0:
        rel_error = np.inf
    else:
        rel_error = abs(avg_shift_excess - theory_shift) / abs(theory_shift)

    return {
        "complexity_power": complexity_power,
        "rel_error": rel_error,
        "measured_shift": avg_shift_excess,
        "theory_shift": theory_shift,
        "n_peri": int(min_indices.size),
        "ok": True,
    }

def golden_search_cp(
    cfg,
    p,
    e,
    n_orbits,
    n_output,
    a,
    b,
    max_iter=25,
    tol=1e-3,
    target_rel_error=None,
):
    """
    Golden-section search on complexity_power in [a, b] to minimize rel_error.

    Parameters
    ----------
    target_rel_error : float or None
        If not None, stop early once best["rel_error"] <= target_rel_error.

    Returns
    -------
    best : dict
        Result dict from measure_precession_error with minimum rel_error.
    n_evals : int
        Number of calls to measure_precession_error.
    """
    phi = (1 + 5**0.5) / 2.0
    invphi = 1.0 / phi

    # Ensure a < b
    if a > b:
        a, b = b, a

    # Initial interior points
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)

    fc = measure_precession_error(cfg, c, n_orbits=n_orbits, n_output=n_output)
    fd = measure_precession_error(cfg, d, n_orbits=n_orbits, n_output=n_output)

    best = fc if fc["rel_error"] <= fd["rel_error"] else fd
    n_evals = 2  # we've evaluated fc and fd

    for _ in range(max_iter):
        # Interval small enough? stop.
        if abs(b - a) < tol:
            break

        if fc["rel_error"] <= fd["rel_error"]:
            # Minimum is in [a, d]; reuse c as new d, compute new c
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = measure_precession_error(cfg, c, n_orbits=n_orbits, n_output=n_output)
            n_evals += 1
        else:
            # Minimum is in [c, b]; reuse d as new c, compute new d
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = measure_precession_error(cfg, d, n_orbits=n_orbits, n_output=n_output)
            n_evals += 1

        # Track best so far
        if fc["rel_error"] < best["rel_error"]:
            best = fc
        if fd["rel_error"] < best["rel_error"]:
            best = fd

        # Early stop if we've hit the requested accuracy
        if (target_rel_error is not None) and (best["rel_error"] <= target_rel_error):
            break

    return best, n_evals



# ======================================================================
#  1D search over complexity_power
# ======================================================================

def calibrate_complexity_power(
    p: float,
    e: float,
    n_orbits: int = 8,
    n_output: int = 12000,
    coarse_range=(0.2, 1.5),
    coarse_steps: int = 13,     # unused now, kept for interface compatibility
    fine_half_width: float = 0.025,
    fine_steps: int = 21,       # unused now, kept for interface compatibility
) -> dict:
    """
    Calibrate complexity_power for a given (p, e):

      1) Configure cfg for (p, e)
      2) Coarse golden search on [coarse_range], stop when rel_error <~ 5%
      3) Fine golden search in a narrow bracket around the coarse best
      4) Return best-fit exponent and stats
    """
    cfg = Config()
    configure_orbit(cfg, p, e)

    print("\n=====================================================")
    print("  Calibrating complexity_power for orbit (p, e)")
    print("=====================================================")
    print(f"  p = {cfg.p:.3f}, e = {cfg.e:.3f}")
    print(f"  k_rq_ratio = {getattr(cfg, 'k_rq_ratio', None)}")
    print(f"  r_q_transition = {cfg.r_q_transition:.3f}")
    print(f"  alpha_gr = {cfg.alpha_gr:.6f}")
    print("-----------------------------------------------------")

    c_min, c_max = coarse_range
    t0 = time.time()

    # ---------------- Coarse golden search ----------------
    print(f"  Coarse search (golden): complexity_power in [{c_min:.2f}, {c_max:.2f}]")
    best_coarse, n_coarse_evals = golden_search_cp(
        cfg,
        p=cfg.p,
        e=cfg.e,
        n_orbits=n_orbits,
        n_output=n_output,
        a=c_min,
        b=c_max,
        max_iter=25,
        tol=0.02,            # coarse resolution in cp
        target_rel_error=0.05,  # stop once we hit ≲5% error
    )

    cp_coarse_best = best_coarse["complexity_power"]
    print("\n  Best coarse exponent:")
    print(
        f"    complexity_power = {cp_coarse_best:.6f}, "
        f"rel_error = {best_coarse['rel_error']*100:.3f}%"
    )

    # ---------------- Fine golden search ----------------
    print("\n  Fine search (golden) around best coarse value...")
    cp_min = max(c_min, cp_coarse_best - fine_half_width)
    cp_max = min(c_max, cp_coarse_best + fine_half_width)

    best_fine, n_fine_evals = golden_search_cp(
        cfg,
        p=cfg.p,
        e=cfg.e,
        n_orbits=n_orbits,
        n_output=n_output,
        a=cp_min,
        b=cp_max,
        max_iter=30,
        tol=1e-3,           # ≈ 0.001 resolution in cp
        target_rel_error=None,  # just run to tol / max_iter
    )

    t1 = time.time()
    total_evals = n_coarse_evals + n_fine_evals
    print(
        f"\n  Calibration completed in {t1 - t0:.2f} s "
        f"({total_evals} simulations)"
    )

    print("\n  ✅ Best-fit complexity exponent:")
    print(
        f"    complexity_power = {best_fine['complexity_power']:.6f}\n"
        f"    rel_error        = {best_fine['rel_error']*100:.3f}%\n"
        f"    measured_shift   = {best_fine['measured_shift']:.6e} rad/orbit\n"
        f"    theory_shift     = {best_fine['theory_shift']:.6e} rad/orbit\n"
        f"    n_peri           = {best_fine['n_peri']}"
    )

    best_fine["ok"] = True
    best_fine["p"] = cfg.p
    best_fine["e"] = cfg.e
    best_fine["k_rq_ratio"] = getattr(cfg, "k_rq_ratio", None)
    best_fine["r_q_transition"] = cfg.r_q_transition

    return best_fine


# ======================================================================
#  Grid scan over (p, e, n_orbits, n_output)
# ======================================================================

def run_grid_scan():
    """
    Run a big calibration scan over multiple (p, e, n_orbits, n_output)
    combinations and save the results to CSV.

    Tweak the grids below as you like.
    """

    # --- You can edit these grids however you want ---
    P_VALUES = [60.0, 120.0]
    E_VALUES = [0.2, 0.4, 0.6, 0.8]
    N_ORBITS_LIST = [4]        # shorter runs for speed
    N_OUTPUT_LIST = [6000, 8000]     # time resolution per run
    # -------------------------------------------------

    combos = list(product(P_VALUES, E_VALUES, N_ORBITS_LIST, N_OUTPUT_LIST))
    total = len(combos)

    print("\n=====================================================")
    print("  GRID SCAN: complexity_power calibration")
    print("=====================================================")
    print(f"  Total combinations: {total}")
    print("  (p, e, n_orbits, n_output) grid as defined in run_grid_scan().")
    print("-----------------------------------------------------")

    records = []

    t_global0 = time.time()
    for idx, (p, e, n_orbits, n_output) in enumerate(combos, start=1):
        print(
            f"\n=== [{idx}/{total}] p={p:.1f}, e={e:.2f}, "
            f"n_orbits={n_orbits}, n_output={n_output} ==="
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

        # Build a flat record for CSV
        record = {
            "p": p,
            "e": e,
            "n_orbits": n_orbits,
            "n_output": n_output,
            "ok": bool(res.get("ok", False)),
            "complexity_power": float(res.get("complexity_power", np.nan)),
            "rel_error": float(res.get("rel_error", np.nan)),
            "measured_shift": float(res.get("measured_shift", np.nan)),
            "theory_shift": float(res.get("theory_shift", np.nan)),
            "n_peri": int(res.get("n_peri", 0)),
            "k_rq_ratio": float(res.get("k_rq_ratio", np.nan)),
            "r_q_transition": float(res.get("r_q_transition", np.nan)),
        }

        print(
            f"  → best cp={record['complexity_power']:.6f}, "
            f"rel_error={record['rel_error']*100:.2f}%, "
            f"ok={record['ok']}"
        )

        records.append(record)

    t_global1 = time.time()
    df = pd.DataFrame(records)
    out_name = "complexity_power_grid_scan.csv"
    df.to_csv(out_name, index=False)

    print("\n=====================================================")
    print("  GRID SCAN COMPLETE")
    print("=====================================================")
    print(f"  Total runtime: {t_global1 - t_global0:.2f} s")
    print(f"  Results saved to: {out_name}")
    print("  Columns:")
    print("    p, e, n_orbits, n_output, ok, complexity_power,")
    print("    rel_error, measured_shift, theory_shift, n_peri,")
    print("    k_rq_ratio, r_q_transition")
    print("=====================================================")

if __name__ == "__main__":
    # MODE 1: single calibration (your original behavior)
    TARGET_P = 100.0
    TARGET_E = 0.6
    N_ORBITS = 8
    N_OUTPUT = 6000

    print("\nRunning single-orbit calibration first...")
    single_result = calibrate_complexity_power(
        p=TARGET_P,
        e=TARGET_E,
        n_orbits=N_ORBITS,
        n_output=N_OUTPUT,
        coarse_range=(0.2, 1.5),
        coarse_steps=11,
        fine_half_width=0.20,
        fine_steps=21,
    )

    if single_result.get("ok", False):
        print(
            f"\nSuggested complexity_power for p={TARGET_P}, e={TARGET_E}: "
            f"{single_result['complexity_power']:.6f}"
        )
    else:
        print("\nSingle calibration failed; no valid measurements.")

    # MODE 2: big grid scan
    print("\n\nNow starting full grid scan...")
    run_grid_scan()
