"""Base initializer: state = initializer(observation)."""

from abc import ABC, abstractmethod

from tomo.state.observation import Observation
from tomo.state.sos_state import SoSState


class Initializer(ABC):
    """Optional module that proposes c0. Does not own final loss."""

    @abstractmethod
    def __call__(self, observation: Observation) -> SoSState:
        """Propose initial state from observation."""
        ...
