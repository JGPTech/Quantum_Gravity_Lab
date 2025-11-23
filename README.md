Fixed calibration issue causing incorrect accuracy. Delete complexity_power_grid_scan.csv and build a fresh one. 


# Quantum Gravity Lab (QGL)

**Quantum Gravity Lab (QGL)** is a small, semi-classical **quantum gravity lab in code**.

It couples:

- a test particle orbiting a compact mass (GR-like orbit),
- a co-moving quantum sensor (Jaynes–Cummings qubit–cavity),
- a phenomenological **complexity scalar** that modulates both gravity and quantum dynamics,

into a single, inspectable simulation pipeline.

The goal is **not** “here is a new theory of quantum gravity.”  
The goal is **“here is a clean, reproducible sandbox where geometry, quantum dynamics, and information-theoretic structure talk to each other.”**

---

## 1. Physics model in one page

### 1.1 Classical sector: orbit in a curved background

QGL tracks a test body of mass `m` moving in the field of a central mass `M`.  
Everything is integrated in coordinate time `t`.

The total acceleration is decomposed as

```text
a_total = a_Newt + a_GR + a_LT
````

where:

#### Newtonian gravity

```text
a_Newt = - G M * r_vec / r^3
```

with `r_vec` the position vector and `r = |r_vec|`.

#### GR-like precession term

We mimic the Schwarzschild 1PN precession by adding an explicit `1 / r^4` force:

```text
a_GR = - Lambda_Q(r) * gamma(v) * r_vec / r^4
```

* `gamma(v) = 1 / sqrt(1 - v^2 / c^2)` is the Lorentz factor.
* `Lambda_Q(r)` is a coupling coefficient,
* `v` is the speed of the test body.

The coupling is

```text
Lambda_Q(r) = alpha_GR * C_Q(r)
alpha_GR    = 3 * G * M * L^2 / c^2
```

where:

* `L` is the orbital angular momentum magnitude,
* `C_Q(r)` is the complexity scalar (see below).

`alpha_GR` is chosen so that, after calibration, the long-term perihelion precession matches the standard GR 1PN value

```text
Δω_GR = 6π G M / (c^2 a (1 - e^2))
```

for a bound orbit with semi-major axis `a` and eccentricity `e`.
This matching is enforced numerically by the **calibration loop** in `calibrate.py`.

#### Lense–Thirring frame dragging (optional)

To represent a spinning central body, QGL can add a simple gravitomagnetic correction:

```text
a_LT = (2 G M a_spin / (c^2 r^3)) * (v_vec × z_hat)
```

where `a_spin ~ J / (M c)` is a dimensionless spin parameter and `z_hat` is the spin axis.
In code this is implemented as an in-plane rotation of the velocity components.

---

### 1.2 Complexity scalar: where “quantum sensitivity” turns on

QGL uses a scalar **complexity field** `C_Q(r)` to control how strongly GR-like and quantum effects are switched on at a given radius:

```text
C_Q(r) = 1 / (1 + (r / r_trans)^p)
r_trans = k_rq * r_apo
r_apo   = a * (1 + e)
```

* `p` (“complexity power”) is a calibrated parameter.
* `r_trans` is tied to the apocenter `r_apo` of the orbit, so the transition scale follows the specific orbit.
* `k_rq` is a dimensionless factor that sets where the transition happens in units of `r_apo`.

Interpretation:

* `C_Q ~ 1` near the compact mass → GR-like corrections and quantum couplings are strong.
* `C_Q ~ 0` far away → dynamics revert to nearly Newtonian and quantum couplings are weak.

The complement

```text
C_N(r) = 1 - C_Q(r)
```

can be used wherever an explicitly “more Newtonian” weighting is convenient.

Operationally, `C_Q(r)` is the **main control knob** that:

* weights GR vs Newtonian forces,
* modulates the Jaynes–Cummings parameters,
* and enters our information-theoretic diagnostics (Bell parameter ansatz, etc).

---

### 1.3 Clocks, lapse, and redshift

The orbit is integrated in **coordinate time** `t` (time at infinity).
The co-moving quantum sensor evolves in **proper time** `τ`.

For a Schwarzschild-like background, the relation is

```text
dτ/dt = sqrt(1 - 2 G M / (c^2 r))
```

This lapse factor is used for:

* converting each coordinate-time step `Δt` into the proper-time step `Δτ` used in quantum evolution:

  ```text
  Δτ = (dτ/dt) * Δt
  ```

* computing simple gravitational redshifts between two radii `r_e` (emitter) and `r_o` (observer):

  ```text
  1 + z = ν_o / ν_e = (dτ/dt at r_e) / (dτ/dt at r_o)
  ```

---

### 1.4 Quantum sector: co-moving Jaynes–Cummings sensor

Along the orbit we propagate a Jaynes–Cummings (JC) system:

* a two-level atom (qubit),
* coupled to a single bosonic mode (cavity).

The Hamiltonian in the rotating-wave approximation is

```text
H_JC = ħ ω_c(r) a† a
     + (ħ ω_a(r) / 2) σ_z
     + ħ g(v) (a† σ_- + a σ_+)
```

The key idea is that **all three parameters** are tied to geometry:

```text
ω_c(r) = ω_c0 * [1 + s_ρ * (ρ(r) / ρ_ref)]
ω_a(r) = ω_a0 * [1 + s_φ * (Φ(r) / Φ_ref)]
g(v)   = g_0  * [1 + s_v * (|v|   / v_ref)]
```

where:

* `ρ(r) ∝ r^-2` is a simple local “density proxy”,
* `Φ(r) = - G M / r` is the gravitational potential,
* `s_ρ`, `s_φ`, `s_v` are user-tunable sensitivity coefficients.

**Time evolution**

At each step:

1. Convert `Δt` → `Δτ` using the lapse factor.

2. Compute the local `H_JC( r(t), v(t) )`.

3. Update the density matrix `ρ` by

   ```text
   ρ(t + Δt) = U ρ(t) U†
   U = exp(-i H_JC Δτ / ħ)
   ```

4. Optionally apply a simple Lindblad-style dissipation for cavity decay and qubit relaxation.

From `ρ` we compute:

* the reduced qubit state `ρ_q` (trace out the cavity),
* Bloch vector components `⟨σ_x⟩`, `⟨σ_y⟩`, `⟨σ_z⟩`,
* photon number `⟨n⟩`,
* coherence `|ρ_01|`,
* and entanglement entropy `S(ρ_q)`.

These are what drive the Bloch sphere and quantum diagnostics in the viewers.

---

### 1.5 Bell-style correlations (phenomenological)

To explore how curvature and motion might affect CHSH-type signals, QGL defines a **model ansatz** for an effective Bell parameter `β(r)`:

```text
β(r) = β_cl + C_Q_eff(r) * (β_qm - β_cl)
```

with:

* `β_cl = 2`   (classical CHSH bound),
* `β_qm = 2√2` (Tsirelson bound).

The effective coefficient combines:

```text
C_Q_eff(r) = C_Q(r)
             * sqrt(1 - 2 G M / (c^2 r))   # gravitational redshift
             * (2 - 1/γ)                   # kinematic factor
```

where `γ` is the Lorentz factor.

**Important:**
This `β(r)` is **explicitly a phenomenological model**, not a claim about the actual behavior of CHSH experiments in GR. The point is to have a tunable, physically-motivated diagnostic that responds to curvature, speed, and complexity in an interpretable way.

---

## 2. Repository layout

```text
QUANTUMGRAVITYLAB/
├── calibrate.py            # Precession-based calibration of complexity_power p, k_rq, etc.
├── documentation.pdf       # Theory & implementation notes (LaTeX/PDF spec)
├── unified_sandbox_core.py # QGL core: config, EOM, complexity field, JC engine
├── unified_simulation.py   # “Offline” QGL run + standard orbit/JC viewer
├── unified_bell_state.py   # Bell State Edition: entanglement + CHSH visualization
├── unified_darkmatter.py   # Dark-matter / protomatter sandbox built on the same core
```

High-level roles:

* **`unified_sandbox_core.py`**
  Core `Config`, orbit integrator, GR/complexity force terms, JC operators, clock conversions, and shared utilities.

* **`unified_simulation.py`**
  A standard **QGL run**: configure an orbit, integrate once, evolve the JC sensor, and inspect everything via a Matplotlib dashboard.

* **`unified_bell_state.py`**
  “Bell State Edition” with:

  * an **Entanglement Console** to choose presets and parameters,
  * a **BellCorrelationViewer** showing `β(t)`, violation regions, entropy, concurrence, discord, orbit coloring, phase space, Bloch sphere, photons, and coherence.

* **`unified_darkmatter.py`**
  Experimental dark-sector / “protomatter” sandbox reusing the same complexity and coupling machinery.

* **`calibrate.py`**
  Scans over `(p, e)` grids, optimizes `complexity_power` to match GR precession, and writes `complexity_power_grid_scan.csv`.
  Main scripts use this table and only fall back to on-the-fly calibration for unseen orbits.

---

## 3. Installation

QGL uses a standard scientific Python stack.

```bash
git clone https://github.com/<your-handle>/QuantumGravityLab.git
cd QuantumGravityLab

python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install numpy scipy matplotlib pandas
```

The reference implementation runs on CPU; array backends can be swapped to CuPy / JAX for GPU experiments.

---

## 4. Quickstart

### 4.1 Standard QGL orbit

```bash
python unified_simulation.py
```

What it does:

* loads or calibrates `complexity_power` for the chosen `(p, e)` orbit,
* integrates the trajectory (including GR-like corrections and optional frame dragging),
* evolves the JC sensor along the orbit in proper time,
* opens an interactive viewer with:

  * orbit geometry,
  * energies and angular momentum,
  * clock rates (GR vs quantum),
  * complexity,
  * Bloch sphere and quantum diagnostics.

### 4.2 Bell State Edition

```bash
python unified_bell_state.py
```

1. **Entanglement Console**

   * choose a preset like “MAX VIOLATION”, “DECOHERENCE”, or “SINGULARITY”,
   * or tune `p`, `e`, `n_orbits`, `n_output`, and JC sensitivity sliders `s_ρ`, `s_φ`, `s_v`.

2. Press **“ENTANGLE”**

   * QGL runs a single high-resolution simulation,
   * launches **BellCorrelationViewer** in a separate process.

3. In the viewer you can:

   * play/pause, scrub time, and change playback speed,
   * see where Bell violations cluster along the orbit,
   * correlate entropy, concurrence, and discord with geometry and complexity,
   * watch the qubit’s Bloch vector trace out trajectories on the sphere,
   * inspect photons, coherence, and complexity over time.

### 4.3 Complexity calibration

```bash
python calibrate.py
```

This script:

* scans a grid of `(p, e)` values,
* finds `complexity_power` for each orbit that matches GR 1PN precession,
* writes results to `complexity_power_grid_scan.csv`.

---

## 5. Scientific status and roadmap

**What QGL *is***:

* A **computational lab** where:

  * GR-inspired corrections,
  * quantum sensors,
  * and information-theoretic diagnostics

  all act on the same orbit in a consistent codebase.

* A platform for experimenting with **geometry → quantum mapping**:
  how curvature and motion can be turned into tunable quantum parameters and observables.

**What QGL *is not***:

* A claim of a complete or correct quantum gravity theory.
* A production-grade astrodynamics package.

### Near-term directions

1. **Fusion-oriented extensions**

   * Replace Keplerian orbits with a simple tokamak-style geometry (closed magnetic field lines).
   * Interpret the “test particle” as a representative ion packet.
   * Use the JC sector as a stylized fusion sensor, and the complexity field as a proxy for local turbulence / instability.
   * Explore how modulation of geometric and complexity parameters might be used as a control-theoretic lever.

2. **Dark matter / protomatter sandbox**

   * Use `unified_darkmatter.py` to test alternative complexity / coupling rules and see how they modify effective mass distributions and orbital signatures.

3. **GPU and ensemble runs**

   * Batch many orbits and JC sensors in parallel on GPU.
   * Turn QGL into a Monte-Carlo-style “quantum gravity lab” for exploring parameter spaces.

---

## 6. License and contact

This project is released under **CC0 1.0 Universal** (public domain dedication).
You are free to copy, modify, and use it for any purpose, including commercial and research work.

Questions, ideas, or collaboration proposals:

**Jon Poplett**
Creator, Quantum Gravity Lab (QGL)

