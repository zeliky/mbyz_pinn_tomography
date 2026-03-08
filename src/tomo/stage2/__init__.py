"""Stage 2: Measurement-aware local refinement policy.

Per-episode, fix one source + its receivers + dynamically selected ROI/support.
Policy outputs small bounded SoS corrections on ROI nodes; environment maps
to 128x128 grid, runs forward operator, computes reward from robust travel-time
residuals with roughness/bounds/step penalties.
"""

from tomo.stage2.roi_builder import build_roi_and_support
from tomo.stage2.stage2_graph import build_stage2_graph
from tomo.stage2.stage2_policy import Stage2Policy
from tomo.stage2.grid_mapper import map_roi_corrections_to_grid
from tomo.stage2.stage2_env import Stage2Env
from tomo.stage2.stage2_objective import stage2_reward
from tomo.stage2.stage2_trainer import Stage2Trainer

__all__ = [
    "build_roi_and_support",
    "build_stage2_graph",
    "Stage2Policy",
    "map_roi_corrections_to_grid",
    "Stage2Env",
    "stage2_reward",
    "Stage2Trainer",
]
