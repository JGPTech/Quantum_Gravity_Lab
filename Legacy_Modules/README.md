# Original applications

These are the original QGL interfaces and calibration tools. “Legacy” identifies their origin, not a deprecation announcement.

| File | Purpose |
| --- | --- |
| `calibrate.py` | Fits complexity exponents to a periapsis-precession target and builds a calibration grid |
| `unified_simulation.py` | Standard orbit and quantum-sensor console/viewer |
| `unified_bell_state.py` | Entanglement console and phenomenological Bell-correlation viewer |
| `unified_darkmatter.py` | Dark-matter-themed orbital/sensor telemetry viewer |

## Connection to the core

The applications use `Core_Framework.unified_sandbox_core` and the calibration module. They also retain viewer-specific quantum and diagnostic calculations. They do not automatically load community modules.

The usual flow is configuration -> calibration -> trajectory -> quantum/diagnostic calculations -> viewer. The Bell application launches its calculation/viewer in a separate Python process.

## Run from the repository root

```sh
python run.py simulation
python run.py bell
python run.py darkmatter
python run.py calibrate
```

Install root `requirements.txt`; interactive use requires a desktop Matplotlib backend. The last command builds the full calibration grid. The launcher keeps calibration CSV files at the repository root.

Package execution is also supported from the root, for example `python -m Legacy_Modules.unified_bell_state --help`. Use the launcher or package entry points rather than executing files inside this folder directly.

The Bell parameter is a model ansatz. The dark-matter name describes the interface and does not establish a dark-matter mechanism. See the [model overview](../Documentation/MODEL.md).
