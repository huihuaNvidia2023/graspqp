"""
Base callback interface.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..problem import OptimizationProblem
    from ..state import TrajectoryState


class Callback(ABC):
    """
    Base class for optimization callbacks.

    Callbacks are called at various points during optimization to enable
    logging, checkpointing, early stopping, etc.
    """

    def on_optimization_start(
        self,
        state: "TrajectoryState",
        problem: "OptimizationProblem",
        n_iters: int,
    ):
        """Called before optimization starts."""
        pass

    def on_optimization_end(
        self,
        state: "TrajectoryState",
        problem: "OptimizationProblem",
        n_iters: int,
    ):
        """Called after optimization ends."""
        pass

    def on_step_start(
        self,
        step: int,
        state: "TrajectoryState",
        problem: "OptimizationProblem",
    ):
        """Called before each optimization step."""
        pass

    def on_step_end(
        self,
        step: int,
        state: "TrajectoryState",
        problem: "OptimizationProblem",
        energies: Dict[str, Any],
    ):
        """Called after each optimization step."""
        pass

    def should_stop(
        self,
        step: int,
        state: "TrajectoryState",
        energies: Dict[str, Any],
    ) -> tuple:
        """
        Check if optimization should stop early.

        Returns:
            (should_stop: bool, reason: str)
        """
        return False, ""


class CallbackList(Callback):
    """Container for multiple callbacks."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def append(self, callback: Callback):
        self.callbacks.append(callback)

    def on_optimization_start(self, state, problem, n_iters):
        for cb in self.callbacks:
            cb.on_optimization_start(state, problem, n_iters)

    def on_optimization_end(self, state, problem, n_iters):
        for cb in self.callbacks:
            cb.on_optimization_end(state, problem, n_iters)

    def on_step_start(self, step, state, problem):
        for cb in self.callbacks:
            cb.on_step_start(step, state, problem)

    def on_step_end(self, step, state, problem, energies):
        for cb in self.callbacks:
            cb.on_step_end(step, state, problem, energies)

    def should_stop(self, step, state, energies) -> tuple:
        for cb in self.callbacks:
            should_stop, reason = cb.should_stop(step, state, energies)
            if should_stop:
                return True, reason
        return False, ""
