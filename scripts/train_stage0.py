"""Stage 0 training entrypoint: train SR initializer (tof_diff_normalized -> 128x128 SOS)."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tomo.training.stage0_initializer import run_stage0


def main() -> None:
    config_dir = str(root / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config_stage0")

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    device_name = cfg.get("device", "cpu")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    model = instantiate(cfg.initializer)
    data_config = OmegaConf.to_container(cfg.data, resolve=True)
    training_config = OmegaConf.to_container(cfg.training, resolve=True)

    best_path = run_stage0(
        model,
        data_config,
        training_config,
        device=device,
    )
    if best_path:
        print(f"Best checkpoint: {best_path}")
    print("Stage 0 training done.")


if __name__ == "__main__":
    main()
