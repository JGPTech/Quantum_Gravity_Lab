# Core framework

The shared engine for the original QGL pipeline. Import it from `Core_Framework.unified_sandbox_core`.

## Contents

`unified_sandbox_core.py` contains:

- `Config`: physical, orbital and quantum-sensor settings.
- Orbit equations, initial conditions and integration helpers.
- Complexity, clock-rate and redshift calculations.
- `JCQuantumEngine`: Jaynes–Cummings quantum evolution.
- `JCComplexityManager` and `ClockComplexityManager`: diagnostic and coupling helpers.

## Connection and data flow

This is the original framework itself. It depends on NumPy, SciPy and pandas, and has no dependency on the viewers or community contributions.

`run_unified_trajectory` integrates an orbit, then optionally evolves quantum diagnostics along it. The quantum output does not feed back into that completed orbit. The separate experimental `run_clock_coupled_trajectory` shares a clock/complexity manager between classical and quantum updates using a simple explicit-Euler integrator.

## Use and extension

Install root `requirements.txt`. A minimal example is in the [root README](../README.md#build-on-the-core). The original interactive applications live in [Legacy_Modules](../Legacy_Modules/README.md).

Contributions should reuse this implementation and document the APIs they call. Changes to shared behavior must be explicit. See the [model overview](../Documentation/MODEL.md) for the meaning and limits of the implemented calculations.
