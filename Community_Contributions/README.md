# Community contributions

Place new work under the contributor's username, then by alignment with the original pipeline. No community implementation is included yet.

```text
Community_Contributions/
└── contributor_username/
    ├── README.md
    ├── __init__.py
    ├── Alignment_01_Direct_Core_Improvements/
    │   └── Component_Name/
    ├── Alignment_02_Original_Roadmap_Extensions/
    │   └── Experiment_Name/
    ├── Alignment_03_Core_Sensor_Based_Extensions/
    │   └── Experiment_Name/
    └── Alignment_04_Independent_Research/
        └── Research_Question/
```

| Alignment | What belongs here |
| --- | --- |
| 01: Direct core improvements | Improvements and tools working directly with the original orbital equations or quantum engine |
| 02: Original roadmap extensions | Applications from the original roadmap, such as a tokamak extension reusing QGL components |
| 03: Core sensor based extensions | New backgrounds or models connected through selected QGL sensor/complexity machinery |
| 04: Independent research | Separate questions such as horizons, dark-energy bookkeeping or tiling, even where an adapter reuses the sensor |

Lower numbers indicate closer alignment, not higher scientific quality. Create only the categories you use. Name importable folders with valid Python identifiers and include `__init__.py` files.

## Documentation and use

Each contributor root, used category and named component needs a README. Use the [README template](../Documentation/README_TEMPLATE.md). Parent READMEs link their children; component READMEs explain their exact core dependencies, data flow, setup, run commands and outputs.

Import the shared core rather than copying it. Document cross-category dependencies. Each component owns its optional requirements and root-relative package command. The original viewers remain independent of community additions unless an explicit integration changes their call paths.

Read [CONTRIBUTING.md](../CONTRIBUTING.md) before submitting.
