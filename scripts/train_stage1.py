"""Stage 1 calibration entrypoint: run Stage 1A and optionally Stage 1B.

Stage and whether to run 1B are controlled via Hydra config/overrides, e.g.:
  python -m scripts.train_stage1
  python -m scripts.train_stage1 run_stage1b=false
  python -m scripts.train_stage1 training=stage1b
"""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tomo.training.stage1_operator_baseline import run_stage1_operator_baseline


def main() -> None:
    config_dir = str(root / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # Default: config_stage1a; overrides from CLI e.g. run_stage1b=false, training=stage1b
        cfg = compose(config_name="config_stage1a", overrides=sys.argv[1:])

    run_stage1b = cfg.get("run_stage1b", True)
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    device_name = cfg.get("device", "cpu")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    full_config = OmegaConf.to_container(cfg, resolve=True)

    summary = run_stage1_operator_baseline(
        full_config,
        device=device,
        max_samples=None,
        run_stage1b=run_stage1b,
    )

    print(f"Processed {summary['num_samples']} samples")
    print(f"Stage 1A avg loss: {summary['stage1a_avg_loss']:.6f}")
    if "stage1b_avg_loss" in summary:
        print(f"Stage 1B avg loss: {summary['stage1b_avg_loss']:.6f}")
    print("Stage 1 calibration done.")


if __name__ == "__main__":
    main()
