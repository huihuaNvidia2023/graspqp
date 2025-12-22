"""
Checkpoint callback for saving intermediate results.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from .base import Callback

if TYPE_CHECKING:
    from ..problem import OptimizationProblem
    from ..state import TrajectoryState


class CheckpointCallback(Callback):
    """
    Callback for saving checkpoints during optimization.

    Args:
        save_every: Save every N steps (default: 500)
        save_dir: Directory to save checkpoints (default: "./checkpoints")
        save_best: Also save the best state so far (default: True)
        prefix: Filename prefix (default: "checkpoint")
    """

    def __init__(
        self,
        save_every: int = 500,
        save_dir: str = "./checkpoints",
        save_best: bool = True,
        prefix: str = "checkpoint",
    ):
        self.save_every = save_every
        self.save_dir = save_dir
        self.save_best = save_best
        self.prefix = prefix

        self._best_energy: Optional[float] = None
        self._best_state: Optional["TrajectoryState"] = None

    def on_optimization_start(self, state, problem, n_iters):
        os.makedirs(self.save_dir, exist_ok=True)
        self._best_energy = None
        self._best_state = None

    def on_step_end(self, step, state, problem, energies):
        # Track best state
        total_energy = sum(e.mean().item() for e in energies.values())

        if self._best_energy is None or total_energy < self._best_energy:
            self._best_energy = total_energy
            self._best_state = state.clone().detach()

        # Save checkpoint
        if step % self.save_every == 0:
            self._save_checkpoint(state, step, energies)

    def on_optimization_end(self, state, problem, n_iters):
        # Save final checkpoint
        self._save_checkpoint(state, n_iters, {}, suffix="_final")

        # Save best checkpoint
        if self.save_best and self._best_state is not None:
            self._save_checkpoint(self._best_state, -1, {}, suffix="_best")

    def _save_checkpoint(
        self,
        state: "TrajectoryState",
        step: int,
        energies: Dict[str, Any],
        suffix: str = "",
    ):
        """Save a checkpoint."""
        if suffix:
            filename = f"{self.prefix}{suffix}.pt"
        else:
            filename = f"{self.prefix}_step_{step}.pt"

        filepath = os.path.join(self.save_dir, filename)

        checkpoint = {
            "step": step,
            "state": state.to_trajectory_dict(),
            "energies": {k: v.detach().cpu() for k, v in energies.items()} if energies else {},
        }

        torch.save(checkpoint, filepath)


class EarlyStoppingCallback(Callback):
    """
    Callback for early stopping based on energy convergence.

    Args:
        patience: Number of steps without improvement before stopping
        min_delta: Minimum change to qualify as improvement
        monitor: Which energy to monitor ("total" or specific cost name)
    """

    def __init__(
        self,
        patience: int = 200,
        min_delta: float = 1e-5,
        monitor: str = "total",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor

        self._best_value: Optional[float] = None
        self._counter: int = 0

    def on_optimization_start(self, state, problem, n_iters):
        self._best_value = None
        self._counter = 0

    def should_stop(self, step, state, energies):
        if self.monitor == "total":
            current = sum(e.mean().item() for e in energies.values())
        else:
            if self.monitor not in energies:
                return False, ""
            current = energies[self.monitor].mean().item()

        if self._best_value is None:
            self._best_value = current
            return False, ""

        if current < self._best_value - self.min_delta:
            self._best_value = current
            self._counter = 0
        else:
            self._counter += 1

        if self._counter >= self.patience:
            return True, f"Early stopping: no improvement for {self.patience} steps"

        return False, ""
