"""Evaluation entrypoint: load config, run system on data, compute metrics."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

# Hydra compose + load checkpoint (optional), run rollout, call metrics
# Stub: print usage until evaluation pipeline is wired.
def main() -> None:
    print("evaluate.py: Load config, run TomographySystem rollout, compute metrics.")
    print("Usage: python -m scripts.evaluate [hydra overrides]")
    print("Config: use same configs/config.yaml as train.")


if __name__ == "__main__":
    main()
