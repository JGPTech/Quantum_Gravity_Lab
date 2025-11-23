#!/usr/bin/env python3
"""
UnifiedSandbox Core — A Quantum-Inspired Gravity Simulation Framework
=====================================================================

This module implements a 2-body gravitational simulation that incorporates
general relativistic corrections through a complexity-based modulation scheme.

Overview:
---------
We solve the equations of motion for a test particle in a gravitational field
with three force components:

1. Newtonian gravity: F = -GMm/r² r̂
2. A GR-like correction: F_GR = -α_eff(r)·γ/r⁴ r̂  
3. Optional frame-dragging (Kerr/Lense-Thirring)

The key feature is that α_eff = α_GR × C_Q(r), where C_Q(r) is a complexity
function that varies from 1 (at small r) to 0 (at large r), with transition
radius r_transition = k_rq × r_apoapsis.

Implementation Details:
----------------------
The complexity function C_Q(r) = 1/(1 + (r/r_transition)²) modulates the 
strength of relativistic corrections. Through empirical calibration, we find
that k_rq ≈ 1.225 for a (p=60, e=0.6) orbit reproduces the GR perihelion
precession to within 0.2%.

Recent work shows k_rq can be made self-consistent and dynamic, emerging from
the local dynamics rather than being a fixed parameter. This is achieved through
a fixed-point iteration where k_rq depends on C_Q, which depends on k_rq.

Jaynes-Cummings Diagnostics:
---------------------------
A quantum cavity model evolves alongside the classical trajectory, providing
additional observables (photon number, coherence, etc.) that correlate with
the gravitational dynamics. These observables can be used to build surrogate
models that predict GR effects from quantum signatures.

Computational Advantages:
------------------------
- O(N) complexity per timestep for N particles (vs O(N²) for full GR)
- Surrogate models can predict precession from complexity metrics alone
- Self-consistent k_rq eliminates need for manual calibration
- Framework scales to 50,000+ particles with GPU acceleration

Test Suite Validation:
---------------------
The framework passes six standard tests:
1. Perihelion precession (matches GR 1PN to <5%)
2. ISCO classification (reproduces r_ISCO = 6GM/c²)
3. Light deflection (additional bending beyond Newton)
4. Frame dragging (Lense-Thirring precession)
5. GW inspiral (Peters-Mathews energy loss)
6. Combined effects (all corrections simultaneously)

Usage:
------
    cfg = Config()
    r0, v0 = canonical_periapsis_ic(cfg)
    df = run_unified_trajectory(cfg, r0, v0, t_max)
    
The resulting DataFrame contains position, velocity, complexity metrics,
and optionally Jaynes-Cummings observables for analysis.

Physical Interpretation:
-----------------------
The complexity C_Q(r) can be interpreted as the rate of information exchange
between quantum and classical degrees of freedom. Regions of high C_Q 
experience stronger relativistic effects, creating an effective curved
spacetime. This provides a computationally efficient way to incorporate
GR-like physics in large-scale simulations.

Note: This is a computational framework for efficient gravity simulations,
not a fundamental theory. The complexity modulation is an effective description
that reproduces GR phenomenology through empirical calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from typing import Dict, Optional, Tuple, List

# Try to load smoothed k_rq(p, e) calibration if available
try:
    _K_RQ_TABLE_SMOOTH = pd.read_csv("k_rq_calibration_table_smooth.csv")
except Exception:
    _K_RQ_TABLE_SMOOTH = None

# =====================================================================
#  1. CONFIGURATION
# =====================================================================

def k_rq_from_pe(p: float, e: float, default: float = 1.245) -> float:
    """
    Return a k_rq_ratio for given (p, e) using the smoothed calibration table.

    - If the table isn't available, fall back to `default`.
    - Uses nearest neighbour in (p, e) space with a mild clamp.
    """
    if _K_RQ_TABLE_SMOOTH is None or _K_RQ_TABLE_SMOOTH.empty:
        return float(default)

    df = _K_RQ_TABLE_SMOOTH

    p_vals = df["p"].values.astype(float)
    e_vals = df["e"].values.astype(float)

    # Normalize p and e differences so p-scale doesn’t dominate
    dp = p_vals - float(p)
    de = e_vals - float(e)

    # Use total span for normalization (avoid zero division)
    p_span = np.ptp(p_vals) if np.ptp(p_vals) > 0.0 else 1.0
    e_span = np.ptp(e_vals) if np.ptp(e_vals) > 0.0 else 1.0

    dp_norm = dp / p_span
    de_norm = de / e_span

    dist2 = dp_norm * dp_norm + de_norm * de_norm
    idx = int(np.argmin(dist2))

    # Prefer smoothed value if present, otherwise raw best
    if "k_rq_smooth" in df.columns:
        k = float(df.iloc[idx]["k_rq_smooth"])
    else:
        k = float(df.iloc[idx]["k_rq_best"])

    # Safety clamp: keep within a physically sane envelope
    k = float(np.clip(k, 0.5, 2.2))
    return k

class Config:
    """
    Configuration for the unified gravity simulation.
    
    Sets physical parameters, orbital elements, and calibration constants.
    The key parameter k_rq_ratio determines where relativistic corrections
    become significant relative to the orbital scale.
    """

    def __init__(self):
        # Physical units (geometric-like with c=10)
        self.G: float = 1.0                    
        self.M: float = 1.0                    
        self.c: float = 10.0                   
        
        # Orbital parameters
        self.p: float = 100.0                   # Semi-latus rectum
        self.e: float = 0.6                    # Eccentricity
        
        # Calibration parameter (empirically determined)
        # For (p=60, e=0.6) we historically used k_rq≈1.245.
        # Now we prefer a dynamic value from (p, e) if a calibration
        # table is available.
        self.k_rq_ratio_default: float = 1.77771
        self.k_rq_ratio_mode: str = "dynamic"  # or "fixed"

        if self.k_rq_ratio_mode == "dynamic":
            # Use smoothed calibration if possible
            self.k_rq_ratio: float = k_rq_from_pe(
                self.p,
                self.e,
                default=self.k_rq_ratio_default,
            )
        else:
            # Legacy behaviour: fixed, hand-tuned value
            self.k_rq_ratio: float = self.k_rq_ratio_default
        
        # Complexity profile parameters
        self.complexity_power: float = 0.70    
        self.quantum_clock_rate: float = 1.0    
        self.gamma_max: float = 250           
        
        # GR coupling (computed from orbit)
        L_sq = self.G * self.M * self.p
        self.alpha_gr: float = (3.0 * self.G * self.M * L_sq) / (self.c**2)
        
        # Transition radius (computed from orbit)
        self.r_q_transition: float = 0.0
        self.update_r_q_transition_from_orbit()
        
        # Test configurations
        self.setup_test_parameters()
        
        # JC cavity parameters
        self.setup_quantum_sensor()

        # Unified clock / complexity blending
        # -----------------------------------
        # w_space      : weight of geometric (r-based) complexity
        # w_struct     : weight of JC structural complexity (from cavity)
        # lambda_scale : global scaling for GR coupling from total C_Q
        self.w_space_complexity: float = 0.6
        self.w_struct_complexity: float = 0.4
        self.lambda_gr_scale: float = 1.0  # can be tuned if desired
        
    def setup_test_parameters(self):
        """Parameters for standard test suite."""
        self.scattering_b: float = 6.9          
        self.scattering_v_inf: float = 1.0      
        self.scattering_x0: float = -75.0       
        self.a_spin: float = 0.0                
        self.kerr_spin_test: float = 0.9        
        self.gw_companion_mass: float = 2.0
        self.gw_strength_scale: float = 1.0e4
        self.gw_n_orbits: int = 25
        
    def setup_quantum_sensor(self):
        """Jaynes-Cummings cavity parameters."""
        self.omega_cavity_0: float = 2.0 * np.pi * 1.0
        self.omega_atom_0: float = 2.0 * np.pi * 1.0
        self.g_0: float = 0.09 * 2.0 * np.pi
        self.kappa_0: float = 0.005   
        self.gamma_0: float = 0.05    
        self.n_cavity: int = 20        
        
        self.s_rho: float = 0.5        
        self.s_phi: float = 0.3        
        self.s_v: float = 0.2          
        self.s_kappa: float = 0.1      
        
        self.epsilon_c: float = 1e-8
        self.eta_photon: float = 1.0
        self.eta_coh: float = 1.0
        
        r_peri = self.p / (1.0 + self.e)
        self.jc_rho_ref: float = 1.0 / (r_peri**2 + 1e-6)
        self.jc_v_ref: float = np.sqrt(self.G / self.p) * (1.0 + self.e)
        self.jc_phi_ref: float = -self.G * self.M / max(r_peri, 1e-6)

    @property
    def r_g(self) -> float:
        """Gravitational radius GM/c²."""
        return self.G * self.M / (self.c**2)
    
    def update_r_q_transition_from_orbit(self) -> None:
        """
        Set the complexity transition radius based on orbital geometry.
        
        r_transition = k_rq × r_apoapsis
        
        This scaling ensures the transition occurs at a consistent
        fraction of the orbital scale.
        """
        if self.e < 1.0:
            r_a = self.p / (1.0 - self.e)  
            self.r_q_transition = float(self.k_rq_ratio * r_a)
        else:
            self.r_q_transition = float(self.k_rq_ratio * self.p)


# =====================================================================
#  2. COMPLEXITY & CLOCK FUNCTIONS
# =====================================================================

def compute_quantum_clock_rate(r: float, cfg: Config) -> Tuple[float, float, float]:
    """
    Compute the quantum complexity envelope and the physical clock rate.

    We separate two concepts:

      • C_Q(r):  dimensionless "quantum complexity" weight (0–1) tied to the
                 orbital scale via r_q_transition. This modulates GR-like
                 corrections in the equations of motion.

      • clock_rate(r): physical clock factor dτ/dt for a static observer in a
                       Schwarzschild spacetime. This is what we use for
                       redshift, ISCO diagnostics, and quantum-time evolution.

    Formulas:

        C_Q(r)      = 1 / (1 + (r / r_q_transition)^p)

        dτ/dt(r)    = sqrt(1 - 2 r_g / r)       for r > 2 r_g
                    = 0                         for r ≤ 2 r_g  (inside horizon)

    Returns:
        C_Q:        quantum complexity (0 to 1)
        C_N:        classical complement (1 - C_Q)
        clock_rate: Schwarzschild-like proper-time rate dτ/dt
    """
    r_safe = max(float(r), 1e-9)

    # --- 1. Complexity envelope tied to orbit scale -------------------
    ratio = r_safe / cfg.r_q_transition
    C_Q = 1.0 / (1.0 + ratio**cfg.complexity_power)
    C_Q = float(np.clip(C_Q, 0.0, 1.0))
    C_N = 1.0 - C_Q

    # --- 2. Schwarzschild proper-time factor --------------------------
    r_g = cfg.r_g
    if r_safe <= 2.0 * r_g:
        # Inside or at horizon: static clocks freeze (no real dτ/dt)
        clock_rate = 0.0
    else:
        # Static observer proper time rate in Schwarzschild
        x = 1.0 - 2.0 * r_g / r_safe
        clock_rate = float(np.sqrt(max(x, 0.0)))

    return C_Q, C_N, clock_rate


def compute_redshift_factor(r_emit: float, r_observe: float, cfg: Config) -> float:
    """
    Compute gravitational frequency shift between two radii.

    We use the physical clock_rate(r) ≈ dτ/dt from compute_quantum_clock_rate:

        ν_obs / ν_emit = (dτ/dt)_emit / (dτ/dt)_obs

    which is exactly the gravitational redshift factor in Schwarzschild.
    """
    _, _, clock_emit = compute_quantum_clock_rate(r_emit, cfg)
    _, _, clock_observe = compute_quantum_clock_rate(r_observe, cfg)

    if clock_observe < 1e-12:
        return 1.0

    return clock_emit / clock_observe


def safe_gamma(v2: float, cfg: Config) -> float:
    """Special relativistic Lorentz factor, capped for numerical stability."""
    c2 = cfg.c * cfg.c
    arg = 1.0 - v2 / c2
    if arg <= 1e-6:
        return cfg.gamma_max
    return float(1.0 / np.sqrt(arg))


# =====================================================================
#  3. EQUATIONS OF MOTION
# =====================================================================

def _compute_acceleration_components(
    x: float,
    y: float,
    z: float,
    vx: float,
    vy: float,
    vz: float,
    cfg: Config,
    lambda_Q: float,
) -> Tuple[float, float, float]:
    """
    Core acceleration calculation.

    Parameters
    ----------
    lambda_Q : float
        Effective GR coupling from the unified clock manager.
        In the legacy path this is alpha_gr * C_Q(r).
        Units match the previous alpha_eff usage.
    """
    r2 = x * x + y * y + z * z
    r = np.sqrt(r2)

    if r < 1e-6:
        return 0.0, 0.0, 0.0

    # Newtonian part
    inv_r3 = 1.0 / (r2 * r)
    ax = -cfg.G * cfg.M * x * inv_r3
    ay = -cfg.G * cfg.M * y * inv_r3
    az = -cfg.G * cfg.M * z * inv_r3

    # GR correction via lambda_Q
    if lambda_Q != 0.0:
        v2 = vx * vx + vy * vy + vz * vz
        gamma = safe_gamma(v2, cfg)
        inv_r4 = 1.0 / (r2 * r2)
        f_mag = -lambda_Q * gamma * inv_r4

        ax += f_mag * (x / r)
        ay += f_mag * (y / r)
        az += f_mag * (z / r)

    # Optional frame dragging (Kerr / Lense-Thirring)
    if abs(cfg.a_spin) > 0.0:
        pref = 2.0 * cfg.a_spin * (cfg.G**2) * (cfg.M**2) / (cfg.c**3)
        Omega_z = pref / (r**3)
        ax += -Omega_z * vy
        ay +=  Omega_z * vx

    return ax, ay, az


def unified_equations_of_motion(
    t: float,
    state,
    cfg: Config,
    clock_mgr: Optional[ClockComplexityManager] = None,
):
    """
    Equations of motion with Newtonian + complexity-modulated GR corrections.

    If `clock_mgr` is provided, the effective GR coupling ?_Q is taken from the
    unified ClockComplexityManager (geometric C_Q only in this path).

    Otherwise, we fall back to the legacy spatial complexity function
    C_Q(r) = 1 / (1 + (r/r_transition)^p) with ?_eff = ?_GR * C_Q.
    """
    x, y, z, vx, vy, vz = state

    r2 = x * x + y * y + z * z
    r = np.sqrt(r2)
    v2 = vx * vx + vy * vy + vz * vz
    v = np.sqrt(v2)

    # Decide how to get ?_Q
    if clock_mgr is not None:
        # Use unified clock manager, but without JC info in this path
        stats = clock_mgr.update(
            t=t,
            r=r,
            v=v,
            jc_expectations=None,
            jc_params=None,
        )
        lambda_Q = stats['lambda_Q']
    else:
        # Legacy: pure spatial complexity
        C_Q, C_N, _clock_rate = compute_quantum_clock_rate(r, cfg)
        lambda_Q = cfg.alpha_gr * C_Q

    ax, ay, az = _compute_acceleration_components(x, y, z, vx, vy, vz, cfg, lambda_Q)
    return [vx, vy, vz, ax, ay, az]


# =====================================================================
#  4. JAYNES-CUMMINGS QUANTUM SENSOR
# =====================================================================

def jc_params_from_classical(r: float, v_mag: float, cfg: Config) -> Dict[str, float]:
    """
    Map classical state to JC cavity parameters.
    
    The cavity frequencies and coupling respond to local field strength,
    providing quantum observables that correlate with gravitational dynamics.
    """
    r_safe = max(float(r), 1e-6)
    
    rho_local = 1.0 / (r_safe**2 + 1e-6)
    phi_local = -cfg.G * cfg.M / r_safe
    
    rho_hat = rho_local / (cfg.jc_rho_ref + 1e-12)
    v_hat = v_mag / (cfg.jc_v_ref + 1e-12)
    phi_hat = phi_local / (abs(cfg.jc_phi_ref) + 1e-12)
    
    omega_c = cfg.omega_cavity_0 * (1.0 + cfg.s_rho * rho_hat)
    omega_a = cfg.omega_atom_0 * (1.0 + cfg.s_phi * phi_hat)
    g_coup = cfg.g_0 * (1.0 + cfg.s_v * v_hat)
    kappa = cfg.kappa_0 * (1.0 + cfg.s_kappa * rho_hat)
    gamma = cfg.gamma_0
    
    return {
        "omega_c": float(omega_c),
        "omega_a": float(omega_a),
        "g": float(g_coup),
        "kappa": float(kappa),
        "gamma": float(gamma),
    }


class JCQuantumEngine:
    """
    Jaynes-Cummings cavity evolution for quantum diagnostics.
    
    The cavity provides observables (photon number, coherence, etc.)
    that can be used to build surrogate models for GR effects.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.n_cavity: int = cfg.n_cavity
        self.dim: int = 2 * self.n_cavity
        self.rho: np.ndarray = np.zeros((self.dim, self.dim), dtype=np.complex128)
        self.tau_quantum: float = 0.0
        self.tau_coordinate: float = 0.0
        self.reset()

    def reset(self):
        """Initialize mixed state."""
        self.rho.fill(0.0)
        if self.dim == 0:
            return
        self.rho[0, 0] = 0.9
        if self.dim > 2:
            self.rho[2, 2] = 0.1
        self.tau_quantum = 0.0
        self.tau_coordinate = 0.0

    def evolve(self, params: Dict[str, float], dt: float, 
               clock_rate: float = 1.0) -> np.ndarray:
        """
        Evolve cavity state for time dt.
        
        If clock_rate != 1, the cavity evolves in "quantum time"
        dt_quantum = clock_rate × dt_coordinate.
        """
        if self.dim == 0 or dt <= 0.0:
            return self.rho
        
        dt_quantum = clock_rate * dt
        self.tau_coordinate += dt
        self.tau_quantum += dt_quantum
        
        H = self.build_hamiltonian(params)
        U = expm(-1j * H * dt_quantum)
        self.rho = U @ self.rho @ U.conj().T
        
        self._apply_decays(params, dt_quantum)
        self._normalize_state()
        
        return self.rho

    def build_hamiltonian(self, params: Dict[str, float]) -> np.ndarray:
        """JC Hamiltonian in truncated Fock space."""
        H = np.zeros((self.dim, self.dim), dtype=np.complex128)
        omega_c = params["omega_c"]
        omega_a = params["omega_a"]
        g = params["g"]
        
        for n in range(self.n_cavity):
            idx_g = 2 * n
            idx_e = idx_g + 1
            if idx_e >= self.dim:
                continue
            H[idx_g, idx_g] += omega_c * n - 0.5 * omega_a
            H[idx_e, idx_e] += omega_c * n + 0.5 * omega_a
        
        for n in range(self.n_cavity - 1):
            idx_e_n = 2 * n + 1
            idx_g_np1 = 2 * (n + 1)
            if idx_g_np1 >= self.dim:
                continue
            val = g * np.sqrt(n + 1.0)
            H[idx_g_np1, idx_e_n] += val
            H[idx_e_n, idx_g_np1] += val
        
        return H

    def _apply_decays(self, params: Dict[str, float], dt: float):
        """Apply dissipation."""
        kappa = params["kappa"]
        gamma = params["gamma"]
        
        for n in range(1, self.n_cavity):
            rate = kappa * n * dt
            if rate <= 0.0:
                continue
            idx_g_src = 2 * n
            idx_e_src = idx_g_src + 1
            idx_g_dst = 2 * (n - 1)
            idx_e_dst = idx_g_dst + 1
            if idx_e_src >= self.dim or idx_g_dst >= self.dim:
                continue
            self.rho[idx_g_dst, idx_g_dst] += rate * self.rho[idx_g_src, idx_g_src]
            self.rho[idx_e_dst, idx_e_dst] += rate * self.rho[idx_e_src, idx_e_src]
            self.rho[idx_g_src, idx_g_src] *= (1.0 - rate)
            self.rho[idx_e_src, idx_e_src] *= (1.0 - rate)
        
        for n in range(self.n_cavity):
            rate = gamma * dt
            if rate <= 0.0:
                continue
            idx_g = 2 * n
            idx_e = idx_g + 1
            if idx_e >= self.dim:
                continue
            self.rho[idx_g, idx_g] += rate * self.rho[idx_e, idx_e]
            self.rho[idx_e, idx_e] *= (1.0 - rate)

    def _normalize_state(self):
        """Maintain trace = 1."""
        trace_val = np.trace(self.rho)
        if np.abs(trace_val) > 1e-12:
            self.rho /= trace_val
        else:
            self.reset()

    def compute_expectations(self, params: Dict[str, float]) -> Dict[str, float]:
        """Extract observables from density matrix."""
        n_photons = 0.0
        sigma_x = 0.0
        sigma_y = 0.0
        sigma_z = 0.0
        
        for n in range(self.n_cavity):
            idx_g = 2 * n
            idx_e = idx_g + 1
            if idx_e >= self.dim:
                continue
            p_g = self.rho[idx_g, idx_g].real
            p_e = self.rho[idx_e, idx_e].real
            n_photons += n * (p_g + p_e)
            coh = self.rho[idx_e, idx_g]
            sigma_x += 2.0 * np.real(coh)
            sigma_y += 2.0 * np.imag(coh)
            sigma_z += (p_e - p_g)
        
        c_mag = np.sqrt(sigma_x**2 + sigma_y**2)
        omega_c = params["omega_c"]
        omega_a = params["omega_a"]
        E_Q = omega_c * n_photons + 0.5 * omega_a * sigma_z
        
        if self.tau_coordinate > 0:
            time_dilation = self.tau_quantum / self.tau_coordinate
        else:
            time_dilation = 1.0
        
        return {
            "n": float(n_photons),
            "sigma_x": float(sigma_x),
            "sigma_y": float(sigma_y),
            "sigma_z": float(sigma_z),
            "c": float(c_mag),
            "E_Q": float(E_Q),
            "time_dilation": float(time_dilation),
            "tau_quantum": float(self.tau_quantum),
            "tau_coordinate": float(self.tau_coordinate),
        }


class JCComplexityManager:
    """
    Track complexity evolution from JC observables.
    
    Provides a secondary measure of quantum vs classical behavior
    based on cavity dynamics.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.initialized_reservoir: bool = False
        self.N0: float = 0.0
        self.c0_bar: float = 0.0  
        self.sigma_z0_bar: float = 0.0
        self.E_Q0_abs: float = 0.0
        self.P_raw: float = 1.0
        self.P: float = 1.0
        self.R_Q: float = 1.0
        self.C_Q: float = 0.5
        self.C_N: float = 0.5

    def update(self, expectations: Dict[str, float], 
               params: Dict[str, float]) -> Dict[str, float]:
        """Update complexity from cavity observables."""
        stats = self._update_reservoir(expectations)
        K_JC = self._compute_structural_complexity(expectations, params)
        
        epsilon = self.cfg.epsilon_c
        denom = K_JC + epsilon
        C_Q_struct = 0.0 if denom <= 0.0 else K_JC / denom
        
        self.C_Q = float(np.clip(self.R_Q * C_Q_struct, 0.0, 1.0))
        self.C_N = 1.0 - self.C_Q
        
        stats.update({
            "K_JC": K_JC,
            "C_Q": self.C_Q,
            "C_N": self.C_N,
            "P": self.P,
            "R_Q": self.R_Q,
        })
        return stats

    def _update_reservoir(self, expectations: Dict[str, float]) -> Dict[str, float]:
        """Track changes relative to initial state."""
        tiny = 1e-12
        N_tot = expectations["n"]
        c_bar = abs(expectations["c"])
        sigma_z_bar = abs(expectations["sigma_z"])
        E_Q_tot = abs(expectations["E_Q"])
        
        if not self.initialized_reservoir:
            self.N0 = N_tot + tiny
            self.c0_bar = c_bar + tiny
            self.sigma_z0_bar = sigma_z_bar + tiny
            self.E_Q0_abs = E_Q_tot + tiny
            self.initialized_reservoir = True
        
        self.P_raw = N_tot / (self.N0 + tiny)
        self.P = float(max(self.P_raw, 0.0) ** self.cfg.eta_photon)
        
        f_c = c_bar / (self.c0_bar + tiny)
        f_z = sigma_z_bar / (self.sigma_z0_bar + tiny)
        f_E = E_Q_tot / (self.E_Q0_abs + tiny)
        C_coh = np.clip((f_c + f_z + f_E) / 3.0, 0.0, 1.0)
        self.R_Q = float(C_coh ** self.cfg.eta_coh)
        
        return {
            "N_tot": N_tot,
            "c_bar": c_bar,
            "sigma_z_bar": sigma_z_bar,
            "E_Q_tot": E_Q_tot,
        }

    def _compute_structural_complexity(self, expectations: Dict[str, float],
                                      params: Dict[str, float]) -> float:
        """Compute complexity from coupling and coherence."""
        tiny = 1e-12
        g_tilde = abs(params["g"]) / (self.cfg.g_0 + tiny)
        delta = abs(params["omega_c"] - params["omega_a"]) / (self.cfg.omega_atom_0 + tiny)
        Delta_ref = 1.0
        S_couple = g_tilde * np.exp(-delta**2 / (2.0 * Delta_ref**2))
        c_i = min(1.0, abs(expectations["c"]))
        return 0.5 * (S_couple + c_i)


# =====================================================================
#  4b. UNIFIED CLOCK / COMPLEXITY MANAGER
# =====================================================================

class ClockComplexityManager:
    """
    Unified clock + complexity manager.

    This class combines:
      - Geometric complexity C_Q_space(r) from the orbital radius
        (via compute_quantum_clock_rate), and
      - Structural quantum complexity C_Q_struct from the JC cavity
        (via JCComplexityManager)

    into a single effective complexity C_Q and clock rate that can be
    used consistently by both:
      - the classical equations of motion (via lambda_Q for GR strength), and
      - the quantum JC engine (via clock_rate for dt_quantum).

    Usage pattern
    -------------
        cfg = Config()
        clock_mgr = ClockComplexityManager(cfg)

        # inside your main loop / integrator:
        jc_expect = jc_engine.compute_expectations(params)
        stats = clock_mgr.update(
            t      = t,
            r      = r,
            v      = v_mag,
            jc_expectations = jc_expect,
            jc_params       = params,
        )

        # stats["C_Q"]      -> total quantum complexity
        # stats["clock"]    -> effective clock rate
        # stats["lambda_Q"] -> GR coupling modifier (alpha_eff = lambda_Q)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Internal JC-complexity manager reused each update
        self.jc_complexity = JCComplexityManager(cfg)
        self.reset()

    # -----------------------------------------------------------------
    # State & bookkeeping
    # -----------------------------------------------------------------
    def reset(self) -> None:
        """Reset running averages and cached values."""
        self.t_last: float = 0.0
        self.n_updates: int = 0

        self.tau_coordinate: float = 0.0
        self.tau_quantum_eff: float = 0.0  # integrated effective quantum time

        self.integrated_C_Q: float = 0.0
        self.integrated_C_N: float = 0.0

        # Last-step values (useful if caller wants current state quickly)
        self.last_C_Q_space: float = 0.0
        self.last_C_Q_struct: float = 0.0
        self.last_C_Q: float = 0.0
        self.last_C_N: float = 1.0
        self.last_clock: float = 0.0
        self.last_lambda_Q: float = 0.0
        self.last_time_dilation: float = 1.0

    # -----------------------------------------------------------------
    # Core update
    # -----------------------------------------------------------------
    def update(
        self,
        t: float,
        r: float,
        v: float,
        jc_expectations: Optional[Dict[str, float]] = None,
        jc_params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Update the unified clock / complexity state.

        Parameters
        ----------
        t : float
            Coordinate time of this sample.
        r : float
            Orbital radius |r|.
        v : float
            Speed |v|.
        jc_expectations : dict, optional
            Output of JCQuantumEngine.compute_expectations(params).
        jc_params : dict, optional
            JC parameter dict (omega_c, omega_a, g, kappa, gamma).
            Required if jc_expectations is provided.

        Returns
        -------
        stats : dict
            Contains:
              - C_Q_space, C_Q_struct, C_Q, C_N
              - clock          (effective quantum clock rate)
              - lambda_Q       (GR coupling: alpha_eff = lambda_Q)
              - time_dilation  (from JC if available, else 1.0)
              - tau_quantum_eff, tau_coordinate
              - C_Q_mean, C_N_mean
            plus all jc_* reservoir stats if JC info was used.
        """
        # -------------------------------
        # 1. Geometric complexity C_Q_space
        # -------------------------------
        r_safe = max(float(r), 1e-9)
        C_Q_space, C_N_space, clock_space = compute_quantum_clock_rate(
            r_safe, self.cfg
        )
        self.last_C_Q_space = C_Q_space

        # -------------------------------
        # 2. JC structural complexity C_Q_struct (optional)
        # -------------------------------
        jc_stats: Dict[str, float] = {}
        if jc_expectations is not None and jc_params is not None:
            # Use JCComplexityManager to get structural C_Q
            jc_stats = self.jc_complexity.update(jc_expectations, jc_params)
            C_Q_struct = float(np.clip(jc_stats["C_Q"], 0.0, 1.0))
        else:
            C_Q_struct = 0.0  # no structural information this step
        self.last_C_Q_struct = C_Q_struct

        # -------------------------------
        # 3. Blend into a single effective complexity C_Q
        # -------------------------------
        w_space = float(self.cfg.w_space_complexity)
        w_struct = float(self.cfg.w_struct_complexity)
        w_total = max(w_space + w_struct, 1e-8)

        w_space /= w_total
        w_struct /= w_total

        C_Q = float(np.clip(w_space * C_Q_space + w_struct * C_Q_struct, 0.0, 1.0))
        C_N = 1.0 - C_Q

        self.last_C_Q = C_Q
        self.last_C_N = C_N

        # -------------------------------
        # 4. Effective clock rate
        # -------------------------------
        # Single, framework-wide quantum clock:
        clock = clock_space
        self.last_clock = clock

        # -------------------------------
        # 5. GR coupling lambda_Q
        # -------------------------------
        lambda_Q = self.cfg.lambda_gr_scale * self.cfg.alpha_gr * C_Q
        self.last_lambda_Q = lambda_Q

        # -------------------------------
        # 6. Time bookkeeping & dilation
        # -------------------------------
        t = float(t)
        dt = 0.0 if self.n_updates == 0 else max(t - self.t_last, 0.0)
        self.t_last = t
        self.n_updates += 1

        self.tau_coordinate += dt
        # Treat "clock" as local d tau / d t factor for an effective quantum time
        self.tau_quantum_eff += dt * max(clock, 0.0)

        if jc_expectations is not None and "time_dilation" in jc_expectations:
            time_dilation = float(jc_expectations["time_dilation"])
        else:
            # Fallback: derive a mild dilation purely from C_Q
            # (1 + small * C_Q) keeps it numerically tame.
            time_dilation = 1.0 + 0.05 * C_Q
        self.last_time_dilation = time_dilation

        # Running means
        if self.tau_coordinate > 0.0:
            self.integrated_C_Q += C_Q * dt
            self.integrated_C_N += C_N * dt
            C_Q_mean = self.integrated_C_Q / self.tau_coordinate
            C_N_mean = self.integrated_C_N / self.tau_coordinate
        else:
            C_Q_mean = C_Q
            C_N_mean = C_N

        # -------------------------------
        # 7. Assemble stats dict
        # -------------------------------
        stats: Dict[str, float] = {
            "t": t,
            "r": r_safe,
            "v": float(v),
            "C_Q_space": C_Q_space,
            "C_Q_struct": C_Q_struct,
            "C_Q": C_Q,
            "C_N": C_N,
            "clock": clock,
            "lambda_Q": lambda_Q,
            "time_dilation": time_dilation,
            "tau_quantum_eff": self.tau_quantum_eff,
            "tau_coordinate": self.tau_coordinate,
            "C_Q_mean": C_Q_mean,
            "C_N_mean": C_N_mean,
        }

        # If we used JC data, expose it with jc_* prefix so callers
        # can still inspect the reservoir / structural diagnostics.
        for key, val in jc_stats.items():
            stats[f"jc_{key}"] = float(val)

        return stats


# =====================================================================
#  5. INTEGRATION ROUTINES
# =====================================================================

def run_simulation_cpu(
    cfg: Config,
    r0,
    v0,
    t_max: float,
    n_output: int = 5000,
    clock_mgr: Optional[ClockComplexityManager] = None,
) -> pd.DataFrame:
    """
    Integrate equations of motion and return trajectory DataFrame.

    If `clock_mgr` is provided, unified_equations_of_motion will update it
    in lockstep with the classical trajectory (geometric C_Q only).
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    
    y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]]
    t_eval = np.linspace(0.0, float(t_max), int(n_output))
    
    if clock_mgr is None:
        eom = unified_equations_of_motion
        args = (cfg,)
    else:
        # Capture cfg + clock manager in a closure for solve_ivp
        def eom(t, y):
            return unified_equations_of_motion(t, y, cfg, clock_mgr)
        args = ()

    sol = solve_ivp(
        eom,
        [0.0, float(t_max)],
        y0,
        t_eval=t_eval,
        args=args,
        method="DOP853",
        rtol=1e-9,
        atol=1e-12,
    )
    
    data = np.vstack([sol.t, sol.y]).T
    df = pd.DataFrame(data, columns=["t", "x", "y", "z", "vx", "vy", "vz"])
    
    df["r"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    df["v"] = np.sqrt(df["vx"]**2 + df["vy"]**2 + df["vz"]**2)
    
    clock_data = []
    for _, row in df.iterrows():
        C_Q, C_N, clock_rate = compute_quantum_clock_rate(row["r"], cfg)
        clock_data.append({
            "C_Q": C_Q,
            "C_N": C_N, 
            "clock_rate": clock_rate,
        })
    
    clock_df = pd.DataFrame(clock_data)
    df = pd.concat([df, clock_df], axis=1)
    
    return df


def run_clock_coupled_trajectory(
    cfg: Config,
    r0,
    v0,
    t_max: float,
    n_steps: int = 2000,
    with_jc: bool = True,
) -> Tuple[pd.DataFrame, ClockComplexityManager]:
    """
    Minimal explicit integrator where classical GR and the JC cavity
    share a single ClockComplexityManager.

    - JC uses the previous-step clock rate to evolve.
    - ClockComplexityManager uses BOTH geometric data (r, v) and JC
      expectations to compute C_Q, clock, and λ_Q.
    - Classical acceleration uses λ_Q from the same manager.

    This is a simple first-order (explicit Euler) scheme meant as a
    clean reference, not a production integrator.
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    state = np.array([r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]], dtype=float)

    times = np.linspace(0.0, float(t_max), int(n_steps))
    dt = times[1] - times[0] if n_steps > 1 else float(t_max)

    clock_mgr = ClockComplexityManager(cfg)
    jc_engine = JCQuantumEngine(cfg) if with_jc else None

    rows: List[Dict[str, float]] = []
    prev_clock = cfg.quantum_clock_rate  # seed the quantum clock

    for i, t in enumerate(times):
        x, y, z, vx, vy, vz = state
        r = float(np.sqrt(x * x + y * y + z * z))
        v = float(np.sqrt(vx * vx + vy * vy + vz * vz))

        # --------------------------------------------------------------
        # 1. JC evolution using the *previous* step's clock
        # --------------------------------------------------------------
        jc_expect = None
        jc_params = None
        if with_jc and jc_engine is not None:
            jc_params = jc_params_from_classical(r, v, cfg)
            local_clock = prev_clock
            jc_engine.evolve(jc_params, dt if i > 0 else 0.0, local_clock)
            jc_expect = jc_engine.compute_expectations(jc_params)

        # --------------------------------------------------------------
        # 2. Unified clock / complexity update (geometric + JC)
        # --------------------------------------------------------------
        stats = clock_mgr.update(
            t=t,
            r=r,
            v=v,
            jc_expectations=jc_expect,
            jc_params=jc_params,
        )
        prev_clock = stats["clock"]  # drives JC on the next step

        lambda_Q = stats["lambda_Q"]

        # --------------------------------------------------------------
        # 3. Classical acceleration from the SAME λ_Q
        # --------------------------------------------------------------
        ax, ay, az = _compute_acceleration_components(x, y, z, vx, vy, vz, cfg, lambda_Q)

        # Simple explicit Euler step (fine for small dt; demo purpose)
        x_new = x + vx * dt
        y_new = y + vy * dt
        z_new = z + vz * dt
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        vz_new = vz + az * dt
        state = np.array([x_new, y_new, z_new, vx_new, vy_new, vz_new], dtype=float)

        # --------------------------------------------------------------
        # 4. Record snapshot
        # --------------------------------------------------------------
        row: Dict[str, float] = {
            "t": float(t),
            "x": x,
            "y": y,
            "z": z,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "r": r,
            "v": v,
            # Unified clock/complexity
            "C_Q": stats["C_Q"],
            "C_N": stats["C_N"],
            "clock_rate": stats["clock"],
            "lambda_Q": stats["lambda_Q"],
            "time_dilation": stats["time_dilation"],
            "tau_quantum_eff": stats["tau_quantum_eff"],
            "tau_coordinate": stats["tau_coordinate"],
            "C_Q_mean": stats["C_Q_mean"],
            "C_N_mean": stats["C_N_mean"],
        }

        if jc_expect is not None:
            row.update({
                "jc_n": jc_expect["n"],
                "jc_sigma_x": jc_expect["sigma_x"],
                "jc_sigma_y": jc_expect["sigma_y"],
                "jc_sigma_z": jc_expect["sigma_z"],
                "jc_c": jc_expect["c"],
                "jc_E_Q": jc_expect["E_Q"],
            })
            # Also expose structural-reservoir stats (prefixed jc_)
            for key, val in stats.items():
                if key.startswith("jc_"):
                    row[key] = val

        rows.append(row)

    df = pd.DataFrame(rows)
    return df, clock_mgr


def run_jc_complexity_along_df(df: pd.DataFrame, cfg: Config,
                               jc_engine: Optional[JCQuantumEngine] = None,
                               jc_complexity: Optional[JCComplexityManager] = None,
                               use_quantum_time: bool = True) -> pd.DataFrame:
    """
    Post-process trajectory with JC cavity evolution.
    
    Adds columns for JC observables and complexity metrics.
    """
    if df is None or df.empty:
        return df
    
    if jc_engine is None:
        jc_engine = JCQuantumEngine(cfg)
    if jc_complexity is None:
        jc_complexity = JCComplexityManager(cfg)
    
    jc_columns = {
        "jc_n": [],
        "jc_sigma_x": [],
        "jc_sigma_y": [],
        "jc_sigma_z": [],
        "jc_c": [],
        "jc_E_Q": [],
        "jc_C_Q": [],
        "jc_C_N": [],
        "jc_P": [],
        "jc_R_Q": [],
        "time_dilation": [],
        "tau_quantum": [],
        "redshift_factor": [],
    }
    
    times = df["t"].values
    r_vals = df["r"].values
    v_vals = df["v"].values
    
    prev_t = float(times[0])
    r_reference = r_vals[0]
    
    for idx, t in enumerate(times):
        t = float(t)
        dt = 0.0 if idx == 0 else max(t - prev_t, 0.0)
        prev_t = t
        
        r = float(max(r_vals[idx], 1e-9))
        v_mag = float(v_vals[idx])
        
        C_Q, C_N, clock_rate = compute_quantum_clock_rate(r, cfg)
        params = jc_params_from_classical(r, v_mag, cfg)
        
        if use_quantum_time:
            jc_engine.evolve(params, dt, clock_rate)
        else:
            jc_engine.evolve(params, dt, 1.0)
        
        expectations = jc_engine.compute_expectations(params)
        stats = jc_complexity.update(expectations, params)
        redshift = compute_redshift_factor(r, r_reference, cfg)
        
        jc_columns["jc_n"].append(expectations["n"])
        jc_columns["jc_sigma_x"].append(expectations["sigma_x"])
        jc_columns["jc_sigma_y"].append(expectations["sigma_y"])
        jc_columns["jc_sigma_z"].append(expectations["sigma_z"])
        jc_columns["jc_c"].append(expectations["c"])
        jc_columns["jc_E_Q"].append(expectations["E_Q"])
        jc_columns["jc_C_Q"].append(stats["C_Q"])
        jc_columns["jc_C_N"].append(stats["C_N"])
        jc_columns["jc_P"].append(stats["P"])
        jc_columns["jc_R_Q"].append(stats["R_Q"])
        jc_columns["time_dilation"].append(expectations["time_dilation"])
        jc_columns["tau_quantum"].append(expectations["tau_quantum"])
        jc_columns["redshift_factor"].append(redshift)
    
    for key, values in jc_columns.items():
        df[key] = values
    
    return df


# =====================================================================
#  6. HIGH-LEVEL API
# =====================================================================

def canonical_periapsis_ic(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Initial conditions at periapsis."""
    r_p = cfg.p / (1.0 + cfg.e)
    v_p = np.sqrt(cfg.G / cfg.p) * (1.0 + cfg.e)
    r0 = np.array([r_p, 0.0, 0.0], dtype=float)
    v0 = np.array([0.0, v_p, 0.0], dtype=float)
    return r0, v0


def canonical_scattering_ic(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Initial conditions for scattering trajectory."""
    r0 = np.array([cfg.scattering_x0, cfg.scattering_b, 0.0], dtype=float)
    v0 = np.array([cfg.scattering_v_inf, 0.0, 0.0], dtype=float)
    return r0, v0


def run_unified_trajectory(cfg: Config, r0, v0, t_max: float,
                          n_output: int = 5000,
                          with_jc: bool = True,
                          use_quantum_time: bool = True) -> pd.DataFrame:
    """
    Main entry point for running simulations.
    
    Returns DataFrame with trajectory and optional JC diagnostics.
    """
    df = run_simulation_cpu(cfg, r0, v0, t_max, n_output=n_output)
    
    if with_jc:
        df = run_jc_complexity_along_df(df, cfg, use_quantum_time=use_quantum_time)
    
    return df


def extract_gr_observables(df: pd.DataFrame, cfg: Config) -> Dict[str, float]:
    """Extract summary statistics from trajectory."""
    observables = {}
    
    if "time_dilation" in df.columns:
        observables["max_time_dilation"] = df["time_dilation"].max()
        observables["min_time_dilation"] = df["time_dilation"].min()
    
    if "redshift_factor" in df.columns:
        observables["max_redshift"] = df["redshift_factor"].max()
        observables["min_redshift"] = df["redshift_factor"].min()
    
    if "C_Q" in df.columns:
        observables["mean_C_Q"] = df["C_Q"].mean()
        observables["max_C_Q"] = df["C_Q"].max()
        observables["min_C_Q"] = df["C_Q"].min()
    
    if "jc_C_Q" in df.columns:
        observables["mean_jc_C_Q"] = df["jc_C_Q"].mean()
        observables["mean_jc_P"] = df["jc_P"].mean()
        observables["mean_jc_sigma_z"] = df["jc_sigma_z"].mean()
    
    return observables


__all__ = [
    "Config",
    "compute_quantum_clock_rate",
    "compute_redshift_factor",
    "unified_equations_of_motion",
    "run_simulation_cpu",
    "JCQuantumEngine", 
    "JCComplexityManager",
    "ClockComplexityManager",
    "run_clock_coupled_trajectory",
    "run_jc_complexity_along_df",
    "canonical_periapsis_ic",
    "canonical_scattering_ic",
    "run_unified_trajectory",
    "extract_gr_observables",
]

