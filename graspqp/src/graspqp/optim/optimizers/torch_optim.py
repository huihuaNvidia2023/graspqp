"""
Optimizers wrapping torch.optim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from torch import Tensor
from torch.optim import LBFGS, SGD, Adam

from .base import Optimizer

if TYPE_CHECKING:
    from ..problem import OptimizationProblem
    from ..state import TrajectoryState


class AdamOptimizer(Optimizer):
    """
    Adam optimizer wrapper.

    Config options:
        lr: Learning rate (default: 0.01)
        betas: Adam betas (default: (0.9, 0.999))
        eps: Epsilon for numerical stability (default: 1e-8)
        weight_decay: L2 regularization (default: 0)
    """

    def __init__(
        self,
        lr: float = 0.01,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def _ensure_optimizer(self, params: List[Tensor]) -> Adam:
        """Create or update internal Adam optimizer."""
        # Always create new optimizer since params are new tensors each step
        self._internal_optimizer = Adam(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        return self._internal_optimizer

    def step(
        self,
        state: "TrajectoryState",
        problem: "OptimizationProblem",
    ) -> "TrajectoryState":
        """Perform one Adam optimization step."""
        # Ensure state requires grad
        state = state.clone()
        state.hand_states.requires_grad_(True)
        state.object_states.requires_grad_(True)

        params = [state.hand_states, state.object_states]
        optimizer = self._ensure_optimizer(params)

        # Zero gradients
        optimizer.zero_grad()

        # Clear step cache
        problem.context.clear_step_cache()

        # Compute total energy
        energy = problem.total_energy(state)
        total = energy.sum()

        # Backward pass
        total.backward()

        # Optimizer step
        optimizer.step()

        self._step_count += 1

        # Return updated state (detached)
        return state.detach()

    def reset(self):
        """Reset optimizer state."""
        super().reset()

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        diag = super().get_diagnostics()
        diag.update(
            {
                "lr": self.lr,
                "betas": self.betas,
            }
        )
        return diag


class SGDOptimizer(Optimizer):
    """
    SGD optimizer wrapper.

    Config options:
        lr: Learning rate (default: 0.01)
        momentum: Momentum factor (default: 0.9)
        weight_decay: L2 regularization (default: 0)
        nesterov: Use Nesterov momentum (default: False)
    """

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0,
        nesterov: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

    def _ensure_optimizer(self, params: List[Tensor]) -> SGD:
        """Create or update internal SGD optimizer."""
        # Always create new optimizer since params are new tensors each step
        self._internal_optimizer = SGD(
            params,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
        )
        return self._internal_optimizer

    def step(
        self,
        state: "TrajectoryState",
        problem: "OptimizationProblem",
    ) -> "TrajectoryState":
        """Perform one SGD optimization step."""
        state = state.clone()
        state.hand_states.requires_grad_(True)
        state.object_states.requires_grad_(True)

        params = [state.hand_states, state.object_states]
        optimizer = self._ensure_optimizer(params)

        optimizer.zero_grad()
        problem.context.clear_step_cache()

        energy = problem.total_energy(state)
        total = energy.sum()
        total.backward()

        optimizer.step()
        self._step_count += 1

        return state.detach()

    def reset(self):
        super().reset()


class LBFGSOptimizer(Optimizer):
    """
    L-BFGS optimizer wrapper.

    Note: L-BFGS requires a closure and may take multiple function evaluations
    per step. Use with caution for large problems.

    Config options:
        lr: Learning rate (default: 1.0)
        max_iter: Max iterations per step (default: 20)
        history_size: History size for L-BFGS (default: 100)
        line_search_fn: Line search function (default: "strong_wolfe")
    """

    def __init__(
        self,
        lr: float = 1.0,
        max_iter: int = 20,
        history_size: int = 100,
        line_search_fn: str = "strong_wolfe",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)
        self.lr = lr
        self.max_iter = max_iter
        self.history_size = history_size
        self.line_search_fn = line_search_fn

    def _ensure_optimizer(self, params: List[Tensor]) -> LBFGS:
        """Create or update internal L-BFGS optimizer."""
        # Always create new optimizer since params are new tensors each step
        self._internal_optimizer = LBFGS(
            params,
            lr=self.lr,
            max_iter=self.max_iter,
            history_size=self.history_size,
            line_search_fn=self.line_search_fn,
        )
        return self._internal_optimizer

    def step(
        self,
        state: "TrajectoryState",
        problem: "OptimizationProblem",
    ) -> "TrajectoryState":
        """Perform one L-BFGS optimization step."""
        state = state.clone()
        state.hand_states.requires_grad_(True)
        state.object_states.requires_grad_(True)

        params = [state.hand_states, state.object_states]
        optimizer = self._ensure_optimizer(params)

        def closure():
            optimizer.zero_grad()
            problem.context.clear_step_cache()
            energy = problem.total_energy(state)
            total = energy.sum()
            total.backward()
            return total

        optimizer.step(closure)
        self._step_count += 1

        return state.detach()

    def reset(self):
        super().reset()
