---

# Quantum Gravity Lab (QGL)

**QGL** is a small, GPU–friendly, semi-classical **quantum gravity lab in code**.

It treats a test particle orbiting a compact mass, a co-moving quantum sensor (Jaynes–Cummings qubit–cavity system), and a phenomenological “complexity scalar” that modulates GR–like corrections to the orbit and the quantum dynamics.

The goal is **not** to claim a new theory of quantum gravity, but to provide a **clean, reproducible sandbox** where:

* Classical GR–like effects (perihelion precession, frame dragging)
* Quantum dynamics (qubit + cavity, entanglement, Bell-type observables)
* Information-theoretic structure (complexity, redshifted clocks)

are all wired into a single, inspectable computational engine.

---

## Physics model in one page

### 1. Classical sector: orbit in a curved background

We model a test body of mass (m) orbiting a central mass (M) in coordinate time (t).

The total acceleration is a superposition of:

[
\mathbf{a}*\text{total}
= \mathbf{a}*\text{Newt}

* \mathbf{a}_\text{GR}
* \mathbf{a}_\text{LT}.
  ]

- **Newtonian gravity**
  [
  \mathbf{a}_\text{Newt} = -\frac{GM}{r^3},\mathbf{r}.
  ]

- **GR-like precession term**
  We emulate the Schwarzschild (1/r^3) correction to the effective potential via an explicit (1/r^4) force term,
  [
  \mathbf{a}*\text{GR}
  = -\Lambda_Q(r),\gamma(v),\frac{\mathbf{r}}{r^4},
  ]
  where (\gamma = (1-v^2/c^2)^{-1/2}).
  The coupling
  [
  \Lambda_Q(r) = \alpha*\text{GR} , C_Q(r), \qquad
  \alpha_\text{GR} = \frac{3GM L^2}{c^2}
  ]
  is chosen so that the long-term perihelion precession matches the standard 1PN GR prediction. This is enforced numerically by the **calibration loop** in `calibrate.py`.

- **Lense–Thirring frame dragging (optional)**
  For a spinning central body with dimensionless spin (a_\text{spin} = J/(Mc)),
  [
  \mathbf{a}*\text{LT}
  = \frac{2GMa*\text{spin}}{c^2 r^3} (\mathbf{v}\times\hat{\mathbf{z}})
  ]
  is added as a gravitomagnetic correction. In code this is implemented via a simple in-plane rotation term.

### 2. Complexity scalar: where GR turns on

QGL uses a scalar **complexity field** (C_Q(r)) to control how “quantum-sensitive” a given region of the orbit is:

[
C_Q(r) = \frac{1}{1 + \left(\dfrac{r}{r_\text{trans}}\right)^p}, \qquad
r_\text{trans} = k_{rq} , a(1+e).
]

* (p) (the “complexity power”) is calibrated from orbit precession.
* (r_\text{trans}) tracks the orbital apocenter, so the transition scale moves with the specific orbit.
* The complement (C_N(r)=1-C_Q(r)) can be used for explicitly Newtonian weighting if desired.

Operationally, **(C_Q) is just a scalar knob** that:

* weights GR vs Newtonian forces,
* modulates quantum sensor parameters,
* and feeds into information-theoretic diagnostics (e.g. Bell parameter ansatz).

### 3. Clocks, lapse, and redshift

The orbit is integrated in coordinate time (t). The co-moving quantum system evolves in proper time (\tau):

[
\frac{d\tau}{dt} = \sqrt{1-\frac{2GM}{c^2 r}}.
]

This same Schwarzschild lapse controls:

* the quantum step size (\Delta\tau = (d\tau/dt),\Delta t),
* and the gravitational redshift used in frequency comparisons:
  [
  1+z = \frac{\nu_o}{\nu_e}
  = \frac{ (d\tau/dt)*{r_e} }{ (d\tau/dt)*{r_o} }.
  ]

### 4. Quantum sector: co-moving JC sensor

Along the orbit we propagate a Jaynes–Cummings (JC) qubit–cavity system with Hamiltonian

[
\hat{H}_\text{JC}
= \hbar\omega_c(r)\hat{a}^\dagger\hat{a}

* \frac{\hbar\omega_a(r)}{2}\hat{\sigma}_z
* \hbar g(v)\left(\hat{a}^\dagger\hat{\sigma}*- + \hat{a}\hat{\sigma}*+\right),
  ]

with **geometry–dependent parameters**:

[
\begin{aligned}
\omega_c(r) &= \omega_{c,0}
\left[1 + s_\rho,\frac{\rho(r)}{\rho_\text{ref}}\right], \
\omega_a(r) &= \omega_{a,0}
\left[1 + s_\phi,\frac{\Phi(r)}{\Phi_\text{ref}}\right], \
g(v) &= g_0
\left[1 + s_v,\frac{|\mathbf{v}|}{v_\text{ref}}\right].
\end{aligned}
]

* (\rho(r)\propto r^{-2}) is a simple local “density proxy”.
* (\Phi(r)=-GM/r) is the Newtonian potential.
* (s_\rho, s_\phi, s_v) control how strongly geometry feeds into the JC sector.

The density matrix (\rho) is evolved by

[
\rho(t+\Delta t)
= e^{-i\hat{H}*\text{JC}\Delta\tau/\hbar},
\rho(t),
e^{+i\hat{H}*\text{JC}\Delta\tau/\hbar},
]

optionally followed by a simple Lindblad-style dissipation step.

From (\rho) we compute:

* the reduced qubit state (\rho_q),
* Bloch vector ((\langle\sigma_x\rangle, \langle\sigma_y\rangle, \langle\sigma_z\rangle)),
* photon number (\langle n\rangle),
* coherence (|\rho_{01}|),
* and entanglement entropy (S(\rho_q)).

### 5. Bell-style correlations (phenomenological)

To explore how curvature and complexity might degrade CHSH-type signals, QGL defines

[
\beta(r)
= \beta_\text{cl}
+ C_Q^\text{eff}(r),\bigl(\beta_\text{qm}-\beta_\text{cl}\bigr),
]

where (\beta_\text{cl}=2) (classical bound) and (\beta_\text{qm}=2\sqrt{2}) (Tsirelson limit). The effective coefficient

[
C_Q^\text{eff}(r)
= C_Q(r),
\sqrt{1-\frac{2GM}{c^2 r}},
\left(2-\frac{1}{\gamma}\right)
]

folds in redshift and relativistic motion. This is **explicitly marked as a model ansatz**, not a derived prediction.

---

## Repository layout

```text
QUANTUMGRAVITYLAB/
├── calibrate.py            # Precession-based calibration of complexity_power p, k_rq, etc.
├── documentation.pdf       # Theory & implementation notes (this spec in LaTeX/PDF form)
├── unified_sandbox_core.py # QGL core: equations of motion, complexity field, JC engine
├── unified_simulation.py   # “Offline” QGL runner + standard orbit/JC dashboards
├── unified_bell_state.py   # Bell State Edition: entanglement + CHSH visualization UI
├── unified_darkmatter.py   # Dark-matter / protomatter sandbox built on the same core
```

High-level roles:

* **`unified_sandbox_core.py`**
  Core `Config`, orbit integrator, GR/complexity terms, JC operators, clock conversions, and helper utilities.

* **`unified_simulation.py`**
  A standard **Quantum Gravity Lab** run: configure orbit, integrate once, and step through the results with interactive Matplotlib dashboards.

* **`unified_bell_state.py`**
  “Bell State Edition” with a front-end **Entanglement Console** and a **BellCorrelationViewer** showing:

  * (\beta(t)) vs CHSH bounds,
  * entanglement entropy, concurrence, and discord,
  * orbit colored by violation strength,
  * phase-space plots,
  * Bloch sphere animation of the qubit.

* **`unified_darkmatter.py`**
  Uses the same machinery to explore dark-sector / protomatter toy models (still experimental).

* **`calibrate.py`**
  Scans over orbits and `complexity_power` values to match measured precession to the GR 1PN value, then writes a `complexity_power_grid_scan.csv` table that the main scripts can load.

---

## Installation

QGL is pure Python + SciPy stack.

```bash
git clone https://github.com/<your-handle>/QuantumGravityLab.git
cd QuantumGravityLab

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install numpy scipy matplotlib pandas
```

(If you want GPU acceleration later, you can swap in CuPy / JAX at the array level, but the reference implementation is CPU-only.)

---

## Quickstart

### 1. Run a standard QGL orbit

```bash
python unified_simulation.py
```

* Choose orbital parameters (p, e, number of orbits, output resolution).
* The script will:

  * look up (or calibrate) `complexity_power` for that orbit,
  * integrate the trajectory,
  * evolve the JC system along it,
  * open a viewer showing orbit, clocks, complexity and Bloch dynamics.

### 2. Bell State Edition

```bash
python unified_bell_state.py
```

* **Stage 1:** Entanglement Console

  * pick a preset (“MAX VIOLATION”, “DECOHERENCE”, “SINGULARITY”),
  * or tune `p`, `e`, `n_orbits`, `n_output`, and JC sensitivity sliders.

* Hit **“ENTANGLE”** to spawn the **BellCorrelationViewer**, which replays a single precomputed run and lets you:

  * scrub time with a slider or play/pause,
  * see where Bell violations cluster in time and along the orbit,
  * watch the qubit’s Bloch vector precess on the sphere,
  * correlate photons, coherence, and entanglement with geometric conditions.

### 3. Calibrate the complexity field explicitly

```bash
python calibrate.py
```

This script (or the calibration helpers inside it) will:

* scan a grid of ((p, e)) values,
* optimize `complexity_power` for each orbit,
* write a `complexity_power_grid_scan.csv` table.

The simulation scripts will read this table and only fall back to on-the-fly calibration when a new ((p,e)) pair is requested.

---

## Scientific status & roadmap

**What QGL is:**

* A **computational sandbox** where classical GR-like corrections, quantum sensors, and information-theoretic diagnostics are all coupled to the same orbit.
* A place to test speculative **mappings** from geometric data → quantum parameters → emergent observables, with full access to the underlying code.

**What QGL is not:**

* A finished quantum gravity theory.
* A precision GR integrator for astrophysical production use.

Current development directions:

1. **Fusion-oriented extensions**

   * Embed a simple **tokamak geometry** into the orbit model (closed field lines instead of Keplerian orbits).
   * Treat the “test particle” as a representative ion packet; promote the JC sector to a stylized fusion sensor.
   * Use the complexity field as a stand-in for local turbulence / instability, and explore control-style modulation of the quantum sector.

2. **Dark matter / protomatter sandbox**

   * Use `unified_darkmatter.py` to explore how alternate complexity / coupling rules change effective mass distributions and orbital signatures.

3. **GPU acceleration and batch ensembles**

   * Run large ensembles of orbits + JC sensors in parallel on GPU, turning QGL into a “Monte Carlo quantum gravity lab”.

---

## License & contact

You are free to read, fork, and experiment with this code.
This project is released under the Creative Commons CC0 1.0 Universal Public Domain Dedication.

For questions, discussion, or collaboration ideas, please open an issue on the repository or contact:

**Jon Poplett**
*Creator, Quantum Gravity Lab*

---
