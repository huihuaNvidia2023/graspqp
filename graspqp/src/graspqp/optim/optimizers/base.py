"""
Base optimizer interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ..problem import OptimizationProblem
    from ..state import TrajectoryState


class Optimizer(ABC):
    """
    Abstract base class for optimizers.

    Optimizers take an optimization problem and state, and produce
    an updated state that (hopefully) has lower total energy.

    Attributes:
        config: Optimizer-specific configuration
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._step_count = 0
        self._internal_optimizer = None

    @property
    def step_count(self) -> int:
        """Number of optimization steps taken."""
        return self._step_count

    @abstractmethod
    def step(
        self,
        state: "TrajectoryState",
        problem: "OptimizationProblem",
    ) -> "TrajectoryState":
        """
        Perform one optimization step.

        Args:
            state: Current trajectory state
            problem: The optimization problem (costs + context)

        Returns:
            Updated trajectory state
        """
        pass

    def reset(self):
        """Reset optimizer state (e.g., momentum buffers)."""
        self._step_count = 0
        self._internal_optimizer = None

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the optimizer."""
        return {
            "step_count": self._step_count,
            "type": self.__class__.__name__,
        }

    def _ensure_optimizer(self, params: list) -> Any:
        """Ensure internal optimizer is initialized."""
        raise NotImplementedError("Subclass must implement _ensure_optimizer")
