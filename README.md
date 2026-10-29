# Quantum Gravity Lab

Quantum Gravity Lab (QGL) is a Python sandbox connecting orbital dynamics, a phenomenological complexity profile, and a Jaynes–Cummings quantum sensor.

Created by Jon Poplett. Released under [CC0](LICENSE.md).

## Project structure

```text
Core_Framework/             Shared simulation engine and configuration
Legacy_Modules/             Original calibration and three interactive applications
Community_Contributions/    Contributor -> alignment -> component
Documentation/             Model overview, original PDFs and README template
External_Dependencies/      Optional external checkout guidance
run.py                     Application launcher
requirements.txt           Python dependencies
CONTRIBUTING.md             How to extend the project
```

## Install

Use Python 3.10 or newer. From the repository root:

```sh
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- Linux/macOS: `source .venv/bin/activate`

Then install the dependencies:

```sh
python -m pip install -r requirements.txt
```

## Run

```sh
python run.py simulation
python run.py bell
python run.py darkmatter
```

These commands open Matplotlib desktop consoles. Select a preset or enter parameters, then launch a simulation. A desktop-capable plotting backend is required.

To build the full calibration grid:

```sh
python run.py calibrate
```

Calibration and quantum evolution can take time. The applications generate `complexity_power_grid_scan.csv` as needed; no calibration table is bundled. The launcher keeps calibration files at the repository root.

## Build on the core

```python
from Core_Framework.unified_sandbox_core import (
    Config, canonical_periapsis_ic, run_unified_trajectory,
)

cfg = Config()
r0, v0 = canonical_periapsis_ic(cfg)
trajectory = run_unified_trajectory(
    cfg, r0, v0, t_max=100.0, n_output=100, with_jc=False,
)
print(trajectory[["t", "r", "C_Q", "clock_rate"]].tail())
```

Read the [core guide](Core_Framework/README.md), [application guide](Legacy_Modules/README.md) and [model overview](Documentation/MODEL.md) for details.

## Contribute

Start with [CONTRIBUTING.md](CONTRIBUTING.md). New work belongs under your username in [Community_Contributions](Community_Contributions/README.md), sorted by its connection to the original pipeline. Each contributor, alignment category and component needs a README using the [documentation template](Documentation/README_TEMPLATE.md).

The shared core is authoritative. Keep proposed changes to its behavior explicit, and document which parts of it your contribution actually uses. Community applications are opt-in; adding one does not automatically change the original viewers.

QGL is an exploratory model, not an established theory of quantum gravity. The Bell signal and geometry-to-sensor mappings are phenomenological. See [Documentation](Documentation/README.md) for further reading.
