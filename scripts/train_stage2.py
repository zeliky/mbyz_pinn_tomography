"""Stage 2 training entrypoint: load Stage 1 outputs, train local refinement policy."""

from __future__ import annotations

import sys
from datetime import datetime
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
from tomo.data.normalization_metadata import resolve_data_config
from tomo.stage2.stage2_env import Stage2Env
from tomo.stage2.stage2_policy import Stage2Policy
from tomo.stage2.stage2_trainer import Stage2Trainer
from tomo.training.stage2_data import load_stage2_samples
from tomo.utils.logging import HtmlTrainingReport, log_stage2_episode_series, setup_training_logging


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
    data_cfg = resolve_data_config(dict(full_cfg.get("data", full_cfg)))
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

    c_base = data_cfg.get("c_base_phys", data_cfg.get("c_base", 1.5))
    c_min = float(train_cfg.get("c_min_phys", train_cfg.get("c_min", 1.45)))
    c_max = float(train_cfg.get("c_max_phys", train_cfg.get("c_max", 1.8)))

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

    enable_html_report = train_cfg.get("enable_html_report", True)
    train_logger = None
    html_report: HtmlTrainingReport | None = None
    if enable_html_report:
        run_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/stage2"))
        log_dir_override = train_cfg.get("log_dir")
        log_root = (
            Path(log_dir_override) / run_id if log_dir_override else ckpt_dir / "runs" / run_id
        )
        train_logger, _ = setup_training_logging(log_root, run_id=run_id)
        html_report = HtmlTrainingReport(log_root / f"output___{run_id}.html")
        html_report.add_text(f"Stage 2  log_dir={log_root}")
        html_report.add_text(f"terminal log: terminal_{run_id}.txt")

    max_episodes = train_cfg.get("max_episodes", 100)
    episode_infos = trainer.train(samples, max_episodes=max_episodes)

    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/stage2"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy_path = checkpoint_dir / "stage2_policy.pt"
    torch.save(policy.state_dict(), policy_path)

    if train_logger is not None and html_report is not None:
        log_stage2_episode_series(train_logger, html_report, episode_infos)
        train_logger.info("Policy saved to %s", policy_path)
    else:
        print(f"Stage 2 training done. {len(episode_infos)} episodes.")
        if episode_infos:
            avg_reward = sum(e.get("reward", 0) for e in episode_infos) / len(episode_infos)
            print(f"Avg episode reward: {avg_reward:.4f}")
        print(f"Policy saved to {policy_path}")


if __name__ == "__main__":
    main()
