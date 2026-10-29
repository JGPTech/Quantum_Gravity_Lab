# Contributing

Welcome! Keep contributions easy to run, clearly attributed, and explicit about their connection to QGL.

## Where code belongs

- `Core_Framework/`: the authoritative shared engine. Discuss substantial behavior changes with the maintainer.
- `Legacy_Modules/`: the original calibration and interactive applications.
- `Community_Contributions/<username>/`: your extensions and experiments, grouped by alignment.

Use the [alignment guide](Community_Contributions/README.md) to choose a category. Create only the folders you need. Place modules, documentation, figures and optional requirements with their owning component.

## Imports and execution

- Import shared APIs from `Core_Framework.unified_sandbox_core`; do not keep a private copy of the core.
- Use normal package imports and add `__init__.py` to importable folders. The `Alignment_01_...` naming keeps folders sorted and valid in Python imports.
- Support commands from the repository root. For example: `python -m Community_Contributions.your_username.Alignment_01_Direct_Core_Improvements.Component.module`.
- Do not manipulate `sys.path` or hard-code personal checkout locations.
- Keep simulations, downloads and window creation out of imports. Use an explicit entry point.
- Document output locations and avoid overwriting shared calibration data or existing results unexpectedly.

## Documentation

Add a README at your contributor root, each used alignment category and each named component. Follow [README_TEMPLATE.md](Documentation/README_TEMPLATE.md).

Explain the purpose, file contents, alignment, exact core APIs used, data flow, setup and run commands. State when a new model generates its own background and merely attaches a QGL sensor. Identify assumptions, limitations, adapted sources and licenses.

Parent READMEs should link their children and explain the relationships between them. Routine figures or support folders can be described in their parent README.

## Dependencies

Keep root requirements limited to the original pipeline. Declare optional dependencies with the contribution. Give installation commands, external source URLs and any required revision or configuration variable. Use `External_Dependencies/` for optional checkouts; do not bundle environments or unexplained third-party code.

## Before submitting

1. Rebase onto the current project structure.
2. Confirm your documented setup and example commands work.
3. Explain any change to core equations, defaults, clocks, decay, calibration or existing outputs. Keep numerical changes distinguishable from file moves.
4. Describe how you checked the change and any remaining limitations. Include focused checks where they help establish behavior; do not add bulk audit artifacts.
5. Complete the PR description and required READMEs. Leave out caches, environments, credentials and unnecessary generated files.

The project uses [CC0](LICENSE.md). Identify authorship and licenses for contributed or adapted material, preserving any third-party notices.
