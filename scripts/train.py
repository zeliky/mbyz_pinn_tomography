"""Single training entrypoint: Hydra + TomographySystem."""

import sys
from pathlib import Path

# Ensure src is on path when running as script
root = Path(__file__).resolve().parents[1]
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch_geometric.data import Data

from tomo.state.observation import Observation
from tomo.systems.tomography_system import TomographySystem


def main() -> None:
    config_dir = str(root / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    device_name = cfg.get("device", "cpu")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    # Build operator and policy from config
    operator = instantiate(cfg.operator)
    policy = instantiate(cfg.policy)

    initializer = None
    if cfg.get("initializer") is not None:
        target = getattr(cfg.initializer, "_target_", None)
        if target not in (None, "null"):
            initializer = instantiate(cfg.initializer)

    training_cfg = cfg.get("training", {})
    rollout_steps = training_cfg.get("rollout_steps", 1)
    alpha = training_cfg.get("alpha", 1.0)

    system = TomographySystem(
        operator=operator,
        policy=policy,
        initializer=initializer,
        rollout_steps=rollout_steps,
        alpha=alpha,
    )

    # Dummy observation and graph for boot
    num_nodes = 10
    tof_obs = torch.zeros(num_nodes, device=device)  # match node count for stub
    observation = Observation(tof_observed=tof_obs)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long, device=device)
    pos = torch.rand(num_nodes, 2, device=device)
    x = torch.full((num_nodes, 1), float("inf"), device=device)
    x[0, 0] = 0.0  # source at node 0
    observation_graph = Data(x=x, edge_index=edge_index, pos=pos, num_nodes=num_nodes)

    state, preds = system.rollout(observation, observation_graph)
    assert state.c_values.shape[0] == num_nodes
    assert len(preds) == rollout_steps
    print("Train boot OK: rollout completed.")

    max_steps = training_cfg.get("max_steps", 1)
    if max_steps > 1:
        from tomo.training.trainer import Trainer
        batch = {
            "observation": observation,
            "observation_graph": observation_graph,
            "tof_observed": tof_obs,
        }
        trainer = Trainer(system, optimizer=None, device=device)
        for step in range(max_steps - 1):
            loss_dict = trainer.train_step(batch)
            if step % 10 == 0:
                print(f"  step {step}: {loss_dict}")
        print("Baseline training steps completed.")


if __name__ == "__main__":
    main()
