# Migration Table (Phase 0 mapping)

## Purpose
This file is the single place where later agents can quickly answer:
1. "Where did a piece of working logic come from?"
2. "Which role in the new unified tomography system does it belong to?"
3. "Should we keep it, adapt it, mine it for logic, or archive it?"

The table is intentionally biased toward paths that currently exist in `src/original/` and toward current duplicated/misaligned code paths under `src/tomo/`.

## Old path -> New role -> Action

| Old path | New role | Action | Notes / discrepancies to watch |
|----------|----------|--------|-----------------------------------|
| `src/original/models/gat.py` (FMMMessagePassing + propagation math) | Operator | Mine + port the "min-style" propagation kernel into `src/tomo/operators/` | Strip training glue and remove node-count assumptions (ex: hard-coded receiver indices). |
| `src/original/models/gat.py` (DualHeadGATModel, SosEstimator) | Operator (core) + training glue | Keep only the propagation math as Operator | Move optimization / losses out of model classes into `losses/*` and orchestration (`TomographySystem`/training). |
| `src/original/RL/env/wave_solver.py` | Operator (FMM forward oracle) | Mine the forward logic; unify with one graph schema | RL variant expects `data.edge_attr` (distances). Ensure `src/tomo/data/graph_builder.py` attaches a consistent `edge_attr`. |
| `src/original/graph/network.py` (GraphDataset, get_graph) | Data / graph schema | Port into `src/tomo/data/graph_builder.py` + optionally `src/tomo/data/transforms.py` | Ensure node ordering and edge attributes match the new operator/policy expectations. |
| `src/original/models/tof_to_sos_net.py`, `resnet_ltsm.py` (TofToSosUNetModel) | Initializer | Port as `src/tomo/initializers/unet_initializer.py` | Initializer should output `SoSState` (or `c0`) only; it must not act as the final solver. |
| `src/original/graph/network.py` (uniform SoS `c_init`) | Initializer | Port as `src/tomo/initializers/constant_init.py` | Constant initializer seeds a baseline SoS state. |
| `src/original/RL/policy/gnn_policy.py` (+ enhanced variant) | Policy | Port as `src/tomo/policies/gnn_policy.py` | Adapt interface to the unified contract: policy produces a delta/update, not final travel times. |
| `src/original/RL/agent.py` (PPO + RolloutBuffer) | Training orchestration | Mine and wrap as `src/tomo/policies/rl_wrapper.py` + trainer integration | Env should talk to `TomographySystem` and not duplicate solver logic. |
| `src/original/main.py` (train scripts) | Orchestration | Archive / consolidate | Replace with a single Hydra entrypoint under `scripts/` that selects modules by config. |
| `src/original/training_steps_handlers.py` | Objective / training | Mine loss logic into `src/tomo/losses/*` and keep training steps in `src/tomo/training/trainer.py` | Avoid embedding losses inside model modules. |
| `src/original/dataset.py` | Data | Port to `src/tomo/data/dataset.py` | Keep dataset output compatible with `Observation` + a single PyG graph schema. |
| `src/original/settings.py` | Config | Replace with Hydra + Pydantic | Move constants (min_sos/max_sos, scaling) into `src/tomo/config/schema.py` and configs. |

## Current codebase duplicates / misaligned paths (update actions)

| Current path | New role | Action |
|--------------|----------|--------|
| `src/tomo/stage2/*` + `scripts/train_stage2.py` | Legacy Stage-2 RL implementation | Archive or integrate later. Do not let this bypass the unified `TomographySystem` loop. |
| `src/tomo/training/stage1_operator_baseline.py` + `src/tomo/training/stage1_search.py` | Not the neural-operator training loop | Treat as a migration-era classical calibration pipeline. Later, port the intended operator-centric baseline into `TomographySystem`-based training. |
| `src/tomo/policies/rl_wrapper.py` vs `src/tomo/stage2/stage2_trainer.py` | PPO/RL training | Deduplicate. Prefer the unified `TomoEnv` + `TomographySystem` approach. |

