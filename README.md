# PowerUp 2026 — Solar PV Analysis (SDM_A vs PPDM_A)

## Overview
This repository contains the implementation used to evaluate and compare the
aggregated single-diode array model (SDM_A) and the per-panel diode model (PPDM_A),
including numerical experiments reported in the accompanying paper.

## Repository layout
- `Solar_PV_stable_250605/`: main project folder used for the study.
- `Solar_PV_stable_250605/LEGACY/`: the exact code path used to generate the paper results.
  The folder name reflects the development timeline only; it is not intended to indicate
  deprecated or unused code.

## Documentation (reproducibility notes)
To keep the codebase concise while ensuring reproducibility, the following procedures
are documented separately:

- **MPPT control** for SDM_A and PPDM_A:
  `Solar_PV_stable_250605/docs/mppt_control.md`

- **Effective average irradiance and temperature** construction used by SDM_A:
  `Solar_PV_stable_250605/docs/effective_avg_irradiance_temperature.md`

These documents provide step-by-step descriptions of the control logic and parameter
construction used in the simulations.
