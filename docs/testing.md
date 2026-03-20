# Testing Status

## Purpose
Help later agents answer:
1. "What tests exist locally?"
2. "Which tests are skipped here and why?"
3. "What command should I run to validate smoke/unit behavior?"

This repo currently has a small set of unit and smoke tests. Some tests depend on:
- The native FMM solver (`py2mat.msfm2d` / `msfm2d`)
- Availability of large MATLAB datasets under `inputData/` (AWS only)

## How to run
- From repo root:
  - `pytest -q`

## Local unit tests (non-data, most should run)
- `tests/unit/test_units.py`
  - Validates SoS unit conversion helpers in `src/tomo/utils/units.py`
  - Includes clamp/roundtrip behavior
- `tests/unit/test_stage0.py`
  - Validates shapes and helper functions for the Stage 0 initializer (SR net)
  - Imports from:
    - `src/tomo/initializers/unet_initializer.py`
    - `src/tomo/training/stage0_initializer.py`
- `tests/unit/test_operator.py`
  - Validates:
    - `GATFMMOperator` output shape and basic ordering behavior
    - `FMMPropagation` determinism and "min-style" propagation sanity
  - Imports from:
    - `src/tomo/operators/gat_fmm_operator.py`
    - `src/tomo/operators/propagation.py`
- `tests/unit/test_stage1_preflight.py`
  - Validates that `run_stage1_operator_baseline()` exits with code `1` when a checkpoint path is missing

## Native FMM-dependent tests (skipped if msfm2d missing)
- `tests/unit/test_forward_tof.py`
  - `forward_tof` shape sanity + finite/positive outputs
  - Guard: tests are skipped if importing `forward_tof` raises a `RuntimeError`
  - Implementation detail: `src/tomo/operators/matlab_fmm_wrapper.py` sets `msfm2d=None` if `py2mat.msfm2d` import fails, and `forward_tof()` raises a `RuntimeError`.

## Smoke tests (subprocess boot)
- `tests/smoke/test_train_boot.py`
  - Runs `python -m scripts.train` as a subprocess
  - Asserts process returns `0` and stdout includes `Train boot OK`

## Dataset-dependent tests (skipped on this machine)
- `tests/smoke/test_dataset.py`
  - Marked skipped: "run on AWS instance with data; cannot run on this computer"
  - Assumes MATLAB `.mat` sources under `inputData/`:
    - `inputData/ForLearning`
    - `inputData/ForValidation`
    - `inputData/ForTest`
    - `inputData/TimeOfFlightData`

## Integration tests gap
- `tests/integration/` directory is currently missing.
- Later agents should consider adding integration tests that:
  - instantiate `TomographySystem` with real lightweight configs
  - run a single end-to-end rollout + objective loss computation using synthetic graphs
  - (optionally) validate that Stage 1/Stage 2 paths do not bypass `TomographySystem` when that constraint becomes mandatory

