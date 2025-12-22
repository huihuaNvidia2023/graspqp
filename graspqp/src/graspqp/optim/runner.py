"""
Optimization runner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from tqdm import tqdm

from .callbacks.base import Callback, CallbackList

if TYPE_CHECKING:
    from .optimizers.base import Optimizer
    from .problem import OptimizationProblem
    from .state import TrajectoryState


class OptimizationRunner:
    """
    Runs the optimization loop with callbacks.

    Handles:
    - Optimization loop with progress bar
    - Callback management
    - Early stopping
    - State management

    Attributes:
        problem: The optimization problem
        optimizer: The optimizer to use
        callbacks: List of callbacks
    """

    def __init__(
        self,
        problem: "OptimizationProblem",
        optimizer: "Optimizer",
        callbacks: Optional[List[Callback]] = None,
    ):
        self.problem = problem
        self.optimizer = optimizer
        self.callbacks = CallbackList(callbacks or [])

    def run(
        self,
        initial_state: "TrajectoryState",
        n_iters: int,
        show_progress: bool = True,
    ) -> "TrajectoryState":
        """
        Run optimization for n_iters steps.

        Args:
            initial_state: Starting trajectory state
            n_iters: Number of optimization iterations
            show_progress: Whether to show tqdm progress bar

        Returns:
            Optimized trajectory state
        """
        state = initial_state.clone()

        # Notify callbacks
        self.callbacks.on_optimization_start(state, self.problem, n_iters)

        # Optimization loop
        iterator = range(1, n_iters + 1)
        if show_progress:
            iterator = tqdm(iterator, desc="Optimizing")

        for step in iterator:
            # Pre-step callback
            self.callbacks.on_step_start(step, state, self.problem)

            # Optimizer step
            state = self.optimizer.step(state, self.problem)

            # Evaluate costs for callbacks
            with torch.no_grad():
                energies = self.problem.evaluate_all(state)

            # Post-step callback
            self.callbacks.on_step_end(step, state, self.problem, energies)

            # Update progress bar
            if show_progress:
                total = sum(e.mean().item() for e in energies.values())
                iterator.set_postfix({"energy": f"{total:.4f}"})

            # Check early stopping
            should_stop, reason = self.callbacks.should_stop(step, state, energies)
            if should_stop:
                if show_progress:
                    print(f"\n{reason}")
                break

        # Notify callbacks
        self.callbacks.on_optimization_end(state, self.problem, n_iters)

        return state

    def step(self, state: "TrajectoryState") -> "TrajectoryState":
        """
        Perform a single optimization step.

        Args:
            state: Current trajectory state

        Returns:
            Updated trajectory state
        """
        return self.optimizer.step(state, self.problem)

    def reset(self):
        """Reset optimizer and callbacks."""
        self.optimizer.reset()
