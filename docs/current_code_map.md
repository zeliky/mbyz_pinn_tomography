# Current Code Map (Phase 0)

Existing logic lives under **src/original/** (git shows deleted `src/*` with counterparts in `src/original/`). No `src/tomo/` existed at plan time; no `pyproject.toml` present.

## Old path → New role → Action

| Old path | New role | Action |
|----------|----------|--------|
| `src/original/models/gat.py` — `FMMMessagePassing`, propagation math | **Operator** | Mine: FMM `aggr='min'`, message `tof_j + dist_ij/speed_j`, `update(aggr_out)`; strip training (SosEstimator's optimizer loop) and node-count assumptions (e.g. `receiver_indices = torch.arange(32, 64)`) |
| `src/original/models/gat.py` — `DualHeadGATModel`, `SosEstimator` | **Operator** (FMM) + training glue | Keep only FMM propagation as operator; move loss/optimization out to Objective/TomographySystem |
| `src/original/RL/env/wave_solver.py` — `FMMMessagePassing`, `BAK_simulate_T` | **Operator** | Mine: same FMM idea; RL variant expects `data.edge_attr` (distances)—unify with single graph schema |
| `src/original/graph/network.py` — `GraphDataset`, `get_graph`, `_build_mesh_edges` | **Data / graph** | Port to `data/graph_builder.py` + `data/transforms.py`; add **edge_attr** (distances) so operator and RL use one schema |
| `src/original/models/tof_to_sos_net.py`, `resnet_ltsm.py` — `TofToSosUNetModel` | **Initializer** | Port as `initializers/unet_initializer.py`; output SoSState/c0 only |
| `src/original/graph/network.py` — `c_init`; `evaluate_gnn_vs_traditional.py` — uniform SoS | **Initializer** | Port as `initializers/constant_init.py` |
| `src/original/RL/policy/gnn_policy.py`, `enhanced_gnn_policy.py` | **Policy** | Port as `policies/gnn_policy.py`; adapt interface to `delta_c = policy(state, observation, tof_error, context)` (output mesh deltas) |
| `src/original/RL/agent.py` — PPO, RolloutBuffer | **Training** | Wrap as `policies/rl_wrapper.py`; env must call TomographySystem, not duplicate solver |
| `src/original/main.py` — 8× `train_*` | **Orchestration** | **Archive**: single entrypoint `scripts/train.py` + Hydra; no competing pipelines |
| `src/original/training_steps_handlers.py` | **Objective / training** | Mine: ToF loss, physics/eikonal, regularization; move into `losses/*` and `training/trainer.py` |
| `src/original/dataset.py` — `TofDataset` | **Data** | Port to `data/dataset.py`; output shape aligned with Observation + graph sample |
| `src/original/settings.py` | **Config** | Replace with Hydra + Pydantic; keep min_sos/max_sos etc. in config/schema |

## Notes

- **Graph schema mismatch**: `src/original/RL/env/wave_solver.py` expects `data.edge_attr` (distances as `[E, 1]`); `GraphDataset.get_graph()` in `network.py` does not set `edge_attr` (only `x`, `edge_index`, `pos`). The new `data/graph_builder.py` (or `edge_features.py`) must compute and attach `edge_attr` so one schema works for both operator and RL.
- **FMM in two places**: `FMMMessagePassing` in `gat.py` holds SoS as `nn.Parameter`; RL `wave_solver.py` passes T, c, pos, edge_index (and edge_attr). The new operator should be stateless: SoS comes from `SoSState`, not from the operator module.

## Audit 2026-03-20 (current `src/tomo` status)

### What is implemented in `src/tomo`
- Operator core exists:
  - `src/tomo/operators/propagation.py` implements `FMMPropagation` as a `MessagePassing` module using min-style aggregation (`aggr="min"`).
  - `src/tomo/operators/gat_fmm_operator.py` implements `GATFMMOperator` as an `Operator` that runs iterative FMM-style propagation driven by SoS extracted from `SoSState`.
- Initializers exist:
  - `src/tomo/initializers/constant_init.py` (constant SoS baseline).
  - `src/tomo/initializers/unet_initializer.py` (SRInitializerNet + UNetInitializer wrapper producing `SoSState`).
- Policies exist:
  - `src/tomo/policies/null_policy.py` (zero delta).
  - `src/tomo/policies/gnn_policy.py` (mesh-node delta updates).
  - `src/tomo/policies/rl_wrapper.py` includes PPO-style wrappers (`TomoEnv`, `RLPolicyAdapter`).
- Unified orchestration exists:
  - `src/tomo/systems/tomography_system.py` implements `TomographySystem` with:
    - `run_one_step()`: operator -> error -> policy -> `SoSState.apply()`
    - `rollout()`: optional initializer -> repeated rollout steps
- Training scaffolding exists:
  - `src/tomo/training/trainer.py` implements `Trainer.train_step()` (rollout -> objective -> backward -> step).

### Known misalignments vs the master plan
1. Stage2 bypasses the unified `TomographySystem` execution flow
   - `src/tomo/stage2/*` and `scripts/train_stage2.py` provide a separate Stage2 RL implementation.
   - `src/tomo/stage2/stage2_env.py` calls `src/tomo/operators/matlab_fmm_wrapper.py::forward_tof()` directly and updates a local `c_current`, without using:
     - `SoSState`
     - `TomographySystem`
     - the unified `operator(state, observation_graph)` contract.

2. Stage1 “operator baseline” is actually derivative-free calibration search
   - `src/tomo/training/stage1_operator_baseline.py` + `src/tomo/training/stage1_search.py` perform derivative-free optimization via `forward_tof()` oracle calls.
   - This is not the intended Phase-4/Phase-6 style “operator-centric” training pipeline under `TomographySystem`.

3. Duplicate policy/trainer concepts for PPO exist
   - Unified RL wrapper: `src/tomo/policies/rl_wrapper.py`
   - Separate Stage2 PPO: `src/tomo/stage2/stage2_policy.py`, `src/tomo/stage2/stage2_trainer.py`

### Immediate guidance for later agents
- Treat the unified contract modules as the source of truth:
  - `src/tomo/systems/tomography_system.py`
  - `src/tomo/operators/*` (operator only)
  - `src/tomo/policies/*` (delta updates)
- Treat `src/tomo/stage2/*` as a legacy/migration-era implementation until it is integrated or archived.
