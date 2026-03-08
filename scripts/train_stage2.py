"""Stage 2 training entrypoint: load Stage 1 outputs, train local refinement policy."""

from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tomo.data.datamodule import TomographyDataModule
from tomo.stage2.stage2_env import Stage2Env, Stage2Sample
from tomo.stage2.stage2_policy import Stage2Policy
from tomo.stage2.stage2_trainer import Stage2Trainer


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_grid_coords(x_s, x_r, grid_shape, coord_range=None):
    H, W = grid_shape
    if coord_range is None:
        return x_s, x_r
    lo, hi = coord_range
    scale = (W - 1) / (hi - lo) if hi != lo else 1.0
    x_s = (x_s - lo) * scale
    x_r = (x_r - lo) * scale
    x_s = np.clip(x_s, 0, W - 1)
    x_r = np.clip(x_r, 0, W - 1)
    return x_s, x_r


def load_stage2_samples(
    stage1_dir: Path,
    dm: TomographyDataModule,
    c_base: float = 1.5,
    c_min: float = 1.45,
    c_max: float = 1.8,
    scale_factor: float = 1.0,
    coord_range: tuple[float, float] | None = None,
    max_samples: int | None = None,
) -> list[Stage2Sample]:
    """Load Stage 1 outputs and build Stage2Sample list."""
    loader = dm.val_dataloader()
    dataset = loader.dataset
    samples = []

    for sample_idx in range(len(dataset)):
        if max_samples is not None and sample_idx >= max_samples:
            break

        s1_dir = stage1_dir / f"sample_{sample_idx}"
        if not s1_dir.exists():
            continue

        best_c_map = np.load(s1_dir / "best_c_map.npy")
        tof_pred = np.load(s1_dir / "tof_pred.npy")

        batch = dataset[sample_idx]
        raw_tof = _to_numpy(batch["raw_tof"]).squeeze()
        x_s = _to_numpy(batch["x_s"]).squeeze()
        x_r = _to_numpy(batch["x_r"]).squeeze()

        H, W = best_c_map.shape
        x_s, x_r = _ensure_grid_coords(x_s, x_r, (H, W), coord_range)

        if x_s.ndim == 1:
            x_s = x_s.reshape(1, -1)
        if x_r.ndim == 1:
            x_r = x_r.reshape(1, -1)

        sample = Stage2Sample(
            c_stage1=best_c_map.astype(np.float64),
            raw_tof=raw_tof.astype(np.float64),
            x_s=x_s.astype(np.float64),
            x_r=x_r.astype(np.float64),
            tof_pred_stage1=tof_pred.astype(np.float64),
            c_base=c_base,
            c_min=c_min,
            c_max=c_max,
            scale_factor=scale_factor,
        )
        samples.append(sample)

    return samples


def main() -> None:
    config_dir = str(root / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config_stage2")

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_name = cfg.get("device", "cpu")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    full_cfg = OmegaConf.to_container(cfg, resolve=True)
    data_cfg = full_cfg.get("data", full_cfg)
    train_cfg = full_cfg.get("training", full_cfg)
    roi_cfg = train_cfg.get("roi", {})
    graph_cfg = train_cfg.get("graph", {})
    policy_cfg = train_cfg.get("policy", {})
    reward_cfg = train_cfg.get("reward", {})
    grid_cfg = train_cfg.get("grid", {})
    episode_cfg = train_cfg.get("episode", {})
    ppo_cfg = train_cfg.get("ppo", {})

    stage1_dir = Path(train_cfg.get("stage1_checkpoint_dir", "checkpoints/stage1b"))
    if not stage1_dir.exists():
        stage1_dir = Path("checkpoints/stage1a")
    if not stage1_dir.exists():
        print("Stage 1 checkpoint dir not found. Run Stage 1 first: python scripts/train_stage1.py")
        sys.exit(1)

    dm = TomographyDataModule(data_cfg)
    dm.setup()

    c_base = data_cfg.get("c_base", 1.5)
    c_min = train_cfg.get("c_min_phys", data_cfg.get("min_sos", 0.1) * 10)
    c_max = train_cfg.get("c_max_phys", data_cfg.get("max_sos", 2.1) * 0.9)
    if c_min > 2:
        c_min = 1.45
    if c_max > 10:
        c_max = 1.8

    scale_factor = train_cfg.get("scale_factor", 1.0)
    coord_range = train_cfg.get("coord_range")
    if coord_range is not None:
        coord_range = tuple(coord_range)

    max_samples = train_cfg.get("max_samples")
    samples = load_stage2_samples(
        stage1_dir,
        dm,
        c_base=c_base,
        c_min=c_min,
        c_max=c_max,
        scale_factor=scale_factor,
        coord_range=coord_range,
        max_samples=max_samples,
    )

    if not samples:
        print("No Stage 2 samples loaded. Ensure Stage 1 checkpoints exist.")
        sys.exit(1)

    print(f"Loaded {len(samples)} Stage 2 samples")

    sample = samples[0]
    env = Stage2Env(
        sample,
        device=device,
        tau_delta=roi_cfg.get("tau_delta", 0.02),
        tau_residual=roi_cfg.get("tau_residual", 0.01),
        dilation_iterations=roi_cfg.get("dilation_iterations", 2),
        support_width=roi_cfg.get("support_width", 1),
        roi_k=graph_cfg.get("roi_k", 20),
        rcv_k=graph_cfg.get("rcv_k", 15),
        spatial_k=graph_cfg.get("spatial_k", 9),
        sigma=grid_cfg.get("sigma", 0.5),
        max_delta=policy_cfg.get("max_delta", 0.02),
        eta=policy_cfg.get("eta", 0.5),
        loss_type=reward_cfg.get("loss_type", "huber"),
        delta=reward_cfg.get("delta", 0.01),
        lambda_smooth=reward_cfg.get("lambda_smooth", 1.0),
        lambda_bounds=reward_cfg.get("lambda_bounds", 10.0),
        lambda_step=reward_cfg.get("lambda_step", 0.1),
        max_steps=episode_cfg.get("max_steps", 5),
        improvement_threshold=episode_cfg.get("improvement_threshold", 1e-5),
    )

    policy = Stage2Policy(
        in_channels=10,
        hidden_channels=policy_cfg.get("hidden_channels", 32),
        num_heads=policy_cfg.get("num_heads", 8),
        max_delta=policy_cfg.get("max_delta", 0.02),
        action_std_init=policy_cfg.get("action_std_init", 0.1),
    )

    trainer = Stage2Trainer(
        policy,
        env,
        lr=ppo_cfg.get("lr", 3e-4),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_eps=ppo_cfg.get("clip_eps", 0.2),
        update_epochs=ppo_cfg.get("update_epochs", 4),
        entropy_coef=ppo_cfg.get("entropy_coef", 0.01),
        value_coef=ppo_cfg.get("value_coef", 0.5),
        rollout_length=ppo_cfg.get("rollout_length", 10),
        device=device,
    )

    max_episodes = train_cfg.get("max_episodes", 100)
    episode_infos = trainer.train(samples, max_episodes=max_episodes)

    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/stage2"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), checkpoint_dir / "stage2_policy.pt")

    print(f"Stage 2 training done. {len(episode_infos)} episodes.")
    if episode_infos:
        avg_reward = sum(e.get("reward", 0) for e in episode_infos) / len(episode_infos)
        print(f"Avg episode reward: {avg_reward:.4f}")
    print(f"Policy saved to {checkpoint_dir / 'stage2_policy.pt'}")


if __name__ == "__main__":
    main()
