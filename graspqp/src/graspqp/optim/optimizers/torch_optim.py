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
    Adam optimizer wrapper with persistent momentum.

    IMPORTANT: This optimizer maintains internal state across steps.
    Call initialize() before the optimization loop to set up persistent parameters.

    Config options:
        lr: Learning rate (default: 0.01)
        betas: Adam betas (default: (0.9, 0.999))
        eps: Epsilon for numerical stability (default: 1e-8)
        weight_decay: L2 regularization (default: 0)
        debug: Enable debug output (default: False)
        min_grad_norm: Minimum gradient norm to prevent vanishing (default: 0, disabled)
    """

    def __init__(
        self,
        lr: float = 0.01,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        debug: bool = False,
        min_grad_norm: float = 0.0,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.debug = debug
        self.min_grad_norm = min_grad_norm

        # Persistent parameters (set by initialize())
        self._hand_param: Optional[Tensor] = None
        self._object_param: Optional[Tensor] = None
        self._internal_optimizer: Optional[Adam] = None

    def initialize(self, state: "TrajectoryState") -> "TrajectoryState":
        """
        Initialize persistent parameters from initial state.
        Must be called before the first step().

        Returns:
            The state with parameters set to the internal persistent tensors.
        """
        # Create persistent parameter tensors
        self._hand_param = state.hand_states.detach().clone().requires_grad_(True)
        self._object_param = state.object_states.detach().clone().requires_grad_(True)

        # Create the Adam optimizer ONCE
        self._internal_optimizer = Adam(
            [self._hand_param, self._object_param],
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

        if self.debug:
            print(
                f"[AdamOptimizer] Initialized with hand_param shape={self._hand_param.shape}, "
                f"object_param shape={self._object_param.shape}"
            )

        # Return state pointing to persistent params
        new_state = state.clone()
        new_state.hand_states = self._hand_param
        new_state.object_states = self._object_param
        return new_state

    def step(
        self,
        state: "TrajectoryState",
        problem: "OptimizationProblem",
    ) -> "TrajectoryState":
        """Perform one Adam optimization step."""
        # If not initialized, do it now (fallback behavior)
        if self._internal_optimizer is None:
            return self._step_create_new(state, problem)

        # Use persistent parameters
        optimizer = self._internal_optimizer

        # Zero gradients
        optimizer.zero_grad()

        # Clear step cache so FK is recomputed with updated params
        problem.context.clear_step_cache()

        # Create state wrapper pointing to persistent params
        current_state = state.clone()
        current_state.hand_states = self._hand_param
        current_state.object_states = self._object_param

        # CRITICAL: Enable gradient mode in context
        # This prevents set_parameters from cloning tensors, preserving gradient flow
        problem.context._skip_set_parameters = True

        try:
            # Compute total energy
            energy = problem.total_energy(current_state)
            total = energy.sum()

            # Backward pass
            total.backward()
        finally:
            # Restore normal mode
            problem.context._skip_set_parameters = False

        # Apply gradient scaling if min_grad_norm is set
        hand_grad = self._hand_param.grad
        if hand_grad is not None and self.min_grad_norm > 0:
            grad_norm = hand_grad.norm()
            if grad_norm < self.min_grad_norm and grad_norm > 1e-10:
                scale = self.min_grad_norm / grad_norm
                self._hand_param.grad = hand_grad * scale
                if self._object_param.grad is not None:
                    self._object_param.grad = self._object_param.grad * scale
                if self.debug and self._step_count % 10 == 0:
                    print(
                        f"  [GradScale] Scaled gradient by {scale:.2f}x (norm {grad_norm:.4f} -> {self.min_grad_norm})"
                    )

        if self.debug and self._step_count % 10 == 0:
            hand_grad = self._hand_param.grad
            if hand_grad is not None:
                grad_norm = hand_grad.norm().item()
                grad_mean = hand_grad.abs().mean().item()
                grad_max = hand_grad.abs().max().item()
                print(
                    f"[AdamOptimizer] Step {self._step_count}: "
                    f"energy={total.item():.4f}, "
                    f"grad_norm={grad_norm:.6f}, "
                    f"grad_mean={grad_mean:.6f}, "
                    f"grad_max={grad_max:.6f}"
                )

                # Check for zero gradients
                if grad_norm < 1e-8:
                    print(f"  WARNING: Near-zero gradient! Optimization may be stuck.")
            else:
                print(f"[AdamOptimizer] Step {self._step_count}: NO GRADIENT!")

        # Optimizer step (updates self._hand_param and self._object_param in-place)
        optimizer.step()

        self._step_count += 1

        # Return state pointing to updated persistent params
        result_state = state.clone()
        result_state.hand_states = self._hand_param.detach().clone().requires_grad_(True)
        result_state.object_states = self._object_param.detach().clone().requires_grad_(True)

        # Update persistent params to the new values (for next step)
        self._hand_param = result_state.hand_states
        self._object_param = result_state.object_states

        # Re-create optimizer with new tensors but copy momentum state
        old_state = optimizer.state_dict()
        self._internal_optimizer = Adam(
            [self._hand_param, self._object_param],
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        # Copy momentum state
        if len(old_state["state"]) > 0:
            new_state_dict = self._internal_optimizer.state_dict()
            # Map old param ids to new ones
            for i, (old_key, old_val) in enumerate(old_state["state"].items()):
                new_key = list(new_state_dict["param_groups"][0]["params"])[i]
                new_state_dict["state"][new_key] = {
                    "step": old_val["step"],
                    "exp_avg": old_val["exp_avg"].clone(),
                    "exp_avg_sq": old_val["exp_avg_sq"].clone(),
                }
            self._internal_optimizer.load_state_dict(new_state_dict)

        return result_state

    def _step_create_new(
        self,
        state: "TrajectoryState",
        problem: "OptimizationProblem",
    ) -> "TrajectoryState":
        """Fallback: create new optimizer each step (original behavior, loses momentum)."""
        state = state.clone()
        state.hand_states.requires_grad_(True)
        state.object_states.requires_grad_(True)

        params = [state.hand_states, state.object_states]
        optimizer = Adam(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

        optimizer.zero_grad()
        problem.context.clear_step_cache()

        # CRITICAL: Enable gradient mode
        problem.context._skip_set_parameters = True
        try:
            energy = problem.total_energy(state)
            total = energy.sum()
            total.backward()
        finally:
            problem.context._skip_set_parameters = False

        if self.debug:
            hand_grad = state.hand_states.grad
            if hand_grad is not None:
                print(
                    f"[AdamOptimizer] Step {self._step_count} (no init): "
                    f"energy={total.item():.4f}, grad_norm={hand_grad.norm().item():.6f}"
                )

        optimizer.step()
        self._step_count += 1

        return state.detach()

    def reset(self):
        """Reset optimizer state."""
        super().reset()
        self._hand_param = None
        self._object_param = None
        self._internal_optimizer = None

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        diag = super().get_diagnostics()
        diag.update(
            {
                "lr": self.lr,
                "betas": self.betas,
                "initialized": self._internal_optimizer is not None,
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
