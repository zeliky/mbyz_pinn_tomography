"""Export predictions (SoS state, ToF) for downstream use."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

# Load config + checkpoint, run system, save state/tof to disk
def main() -> None:
    print("export_predictions.py: Run system, save state and ToF predictions.")
    print("Usage: python -m scripts.export_predictions output_dir=... [overrides]")


if __name__ == "__main__":
    main()
