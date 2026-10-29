# Contributor README template

Add a README to your contributor folder, each used alignment category and each named component. Keep it practical: a new contributor should understand the connection to QGL and run a small example without reading the implementation first.

## Component README

Use these sections and replace the instructions with specific details:

```markdown
# Component name

## Purpose
What question does this address, and what does it produce?

## Contents
List the modules and explain their roles. Identify figures, data and configuration.

## Alignment with QGL
Name the alignment category and explain the connection to the original project.

## Core connection and data flow
Name the core classes/functions actually used. Distinguish using the original
equations, inheriting the engine, importing configuration and adapting formulas.
State explicitly when there is no direct core dependency.
Describe inputs -> background/trajectory -> sensor or analysis -> outputs.
State whether sensor feedback changes the background.

## Setup and running
Give installation and example commands from the repository root.
List optional dependencies, external source URLs/revisions and configuration.
Specify expected outputs, output locations and any desktop requirements.

## Assumptions and limitations
Explain approximations, toy mappings, fitted parameters and known limits.
Briefly describe how to check the example and any unresolved issues.

## Effect on the original pipeline
State whether this changes the core, legacy applications, defaults or outputs.
Distinguish opt-in alternatives from behavior already adopted by the project.

## Authors and references
Credit contributors, adapted code, papers, data and licenses.
Distinguish reproductions from independently constructed examples.
```

## Parent README

For contributor roots and alignment categories, include:

1. Purpose and placement in the contributor -> alignment -> component chain.
2. An inventory linking each child README and describing its role.
3. An explanation of alignment and any cross-component dependencies.
4. Setup/run entry points, or links to the component instructions.
5. Authors and source/license credits.

Routine support folders can be explained in their parent README. Use a separate README when a dataset, tool or experiment has its own setup or purpose. If a component becomes a separate repository, use the component format at its root and link its relationship back to QGL.
