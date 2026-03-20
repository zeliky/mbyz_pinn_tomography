# Known Dependencies

## Purpose
Document the non-Python/native dependencies and how they affect which parts of the repo can run.
This is designed so later agents can quickly predict:
1. why a test is skipped
2. which runtime paths will fail without native libs

## Native FMM solver (`py2mat.msfm2d`)

### Where it is loaded
- `src/tomo/operators/matlab_fmm_wrapper.py`
  - Tries `from py2mat.msfm2d import msfm2d`
  - If import fails, it sets `msfm2d = None`
  - `forward_tof()` raises a `RuntimeError` if `msfm2d is None`

### Where it is called in the current codebase
- `src/tomo/training/stage1_operator_baseline.py`
  - `_run_stage1_preflight()` calls `forward_tof()` as an oracle sanity check
- `src/tomo/stage2/stage2_env.py`
  - `Stage2Env.step()` calls `forward_tof()` to evaluate travel-time predictions for the refined SoS grid
- `tests/unit/test_forward_tof.py`
  - The entire test file is skipped if `forward_tof` import/availability fails

### CI / local implications
- Without `py2mat.msfm2d` available:
  - Stage 1 and Stage 2 (legacy implementations) will fail where they call `forward_tof()`
  - `tests/unit/test_forward_tof.py` will be skipped
- The unified operator path (non-C) is implemented in:
  - `src/tomo/operators/propagation.py` (message passing)
  - `src/tomo/operators/gat_fmm_operator.py` (iterative FMM-style propagation)

## PyTorch Geometric (PyG)

The operator/policy stack depends on PyG:
- `src/tomo/operators/propagation.py`: `MessagePassing`
- `src/tomo/policies/gnn_policy.py`: uses `GATConv`
- Graph containers are PyG `torch_geometric.data.Data`

If PyG is missing/broken, the operator and policy won't import.

