"""Policies: learned optimizer, proposes delta_c."""

from tomo.policies.base import Policy
from tomo.policies.null_policy import NullPolicy

__all__ = ["Policy", "NullPolicy"]
