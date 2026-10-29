# Model overview

## Orbit and complexity

QGL adds a complexity-weighted, GR-inspired radial correction and optional in-plane frame dragging to Newtonian acceleration. The complexity profile is:

```text
C_Q(r) = 1 / (1 + (r / r_transition)^complexity_power)
```

The transition radius follows the orbital scale. Calibration adjusts the complexity exponent to match a target periapsis advance. This is a phenomenological model; a calibrated match is not an independent prediction.

## Clocks and quantum sensor

`compute_quantum_clock_rate` returns `(C_Q, C_N, clock_rate)`. The clock factor uses the static Schwarzschild lapse outside the horizon; it is not a complete moving-observer clock model.

The Jaynes–Cummings engine evolves a truncated qubit–cavity density matrix. Geometry, potential and speed enter chosen sensor parameters. The coupling is a modeling choice. The original decay step can produce a non-positive density matrix; an alternative implementation would need an explicit core change or opt-in extension.

## Execution paths

- `run_unified_trajectory`: orbit first, then optional quantum diagnostics along the result.
- `run_clock_coupled_trajectory`: a separate experimental path with shared classical/quantum coupling and a simple explicit-Euler integrator.
- Original viewers: use the core and add their own quantum and display calculations.

## Interpretation

The Bell parameter is a phenomenological ansatz, and the dark-matter application is primarily a themed viewer. Calibration uses sampled periapsis minima and depends on output resolution. Frame dragging is an approximate term. The framework is intended for exploration and extension, not as an established quantum-gravity theory or a production astrodynamics package.

When extending QGL, state which original calculations you reuse, which equations you replace, and whether the sensor feeds back into its background.
