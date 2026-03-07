"""Stage 1 calibration entrypoint: run Stage 1A and Stage 1B."""

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
    config_name = "config_stage1a"
    run_1b = True
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("stage1a", "1a"):
            config_name = "config_stage1a"
            run_1b = False
        elif arg in ("stage1b", "1b"):
            config_name = "config_stage1b"
            run_1b = True

    config_dir = str(root / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    device_name = cfg.get("device", "cpu")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    full_config = OmegaConf.to_container(cfg, resolve=True)

    summary = run_stage1_operator_baseline(
        full_config,
        device=device,
        max_samples=None,
        run_stage1b=run_1b,
    )

    print(f"Processed {summary['num_samples']} samples")
    print(f"Stage 1A avg loss: {summary['stage1a_avg_loss']:.6f}")
    if "stage1b_avg_loss" in summary:
        print(f"Stage 1B avg loss: {summary['stage1b_avg_loss']:.6f}")
    print("Stage 1 calibration done.")


if __name__ == "__main__":
    main()
