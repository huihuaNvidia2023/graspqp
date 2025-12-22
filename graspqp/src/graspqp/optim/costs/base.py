"""
Base classes for cost functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ..context import OptimizationContext
    from ..state import TrajectoryState


@dataclass
class CostConfig:
    """Base configuration for cost functions."""

    weight: float = 1.0
    enabled: bool = True

    # Additional config can be added in subclasses
    extra: Dict[str, Any] = field(default_factory=dict)


class CostFunction(ABC):
    """
    Abstract base class for all cost functions.

    Cost functions compute energy values that the optimizer tries to minimize.
    They can be per-frame (independent computation per frame) or temporal
    (requiring multiple frames).

    Attributes:
        name: Unique identifier for this cost
        weight: Multiplier applied to the cost value
        enabled: Whether this cost is active
        config: Cost-specific configuration
    """

    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.weight = weight
        self.enabled = enabled
        self.config = config or {}

    @property
    def is_temporal(self) -> bool:
        """Whether this cost requires multiple frames."""
        return False

    @property
    def required_cache_keys(self) -> List[str]:
        """Cache keys this cost depends on."""
        return []

    @property
    def provided_cache_keys(self) -> List[str]:
        """Cache keys this cost provides."""
        return []

    @abstractmethod
    def evaluate(
        self,
        state: "TrajectoryState",
        ctx: "OptimizationContext",
    ) -> Tensor:
        """
        Compute the cost value.

        Args:
            state: Current trajectory state
            ctx: Optimization context with models and cache

        Returns:
            Cost values per trajectory. Shape: (B,) or (B*K,) if flattened
        """
        pass

    def __call__(
        self,
        state: "TrajectoryState",
        ctx: "OptimizationContext",
    ) -> Tensor:
        """Evaluate cost with weight applied."""
        if not self.enabled:
            # Return zeros
            if state._has_perturbations:
                return torch.zeros(state.B * state.K, device=state.device)
            return torch.zeros(state.B, device=state.device)

        return self.weight * self.evaluate(state, ctx)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CostFunction":
        """Create cost from configuration dict."""
        return cls(
            name=config.get("name", cls.__name__),
            weight=config.get("weight", 1.0),
            enabled=config.get("enabled", True),
            config=config.get("config", {}),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, weight={self.weight}, enabled={self.enabled})"


class PerFrameCost(CostFunction):
    """
    Base class for costs that operate on individual frames independently.

    The evaluate() method processes all frames at once in a vectorized manner,
    then aggregates the per-frame costs into per-trajectory costs.
    """

    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
        aggregation: str = "sum",
    ):
        super().__init__(name, weight, enabled, config)
        self.aggregation = aggregation  # "sum", "mean", "max"

    @abstractmethod
    def evaluate_frames(
        self,
        state: "TrajectoryState",
        ctx: "OptimizationContext",
    ) -> Tensor:
        """
        Compute per-frame cost values.

        Args:
            state: Trajectory state (already flattened if has perturbations)
            ctx: Optimization context

        Returns:
            Per-frame costs. Shape: (B, T) or (B*K, T)
        """
        pass

    def evaluate(
        self,
        state: "TrajectoryState",
        ctx: "OptimizationContext",
    ) -> Tensor:
        """
        Evaluate cost, aggregating per-frame values.

        Returns:
            Per-trajectory costs. Shape: (B,) or (B*K,)
        """
        # Flatten perturbations for computation
        if state._has_perturbations:
            flat_state = state.flatten_perturbations()
        else:
            flat_state = state

        # Get per-frame costs: (B*K, T) or (B, T)
        per_frame = self.evaluate_frames(flat_state, ctx)

        # Apply valid mask
        if flat_state.valid_mask is not None:
            per_frame = per_frame * flat_state.valid_mask.float()

        # Aggregate across time
        if self.aggregation == "sum":
            return per_frame.sum(dim=-1)
        elif self.aggregation == "mean":
            lengths = flat_state.lengths.float().clamp(min=1)
            return per_frame.sum(dim=-1) / lengths
        elif self.aggregation == "max":
            return per_frame.max(dim=-1).values
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


class TemporalCost(CostFunction):
    """
    Base class for costs that require multiple frames (velocity, acceleration, etc.).

    Temporal costs operate on frame differences or windows.
    """

    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
        window_size: int = 2,
    ):
        super().__init__(name, weight, enabled, config)
        self.window_size = window_size

    @property
    def is_temporal(self) -> bool:
        return True

    @abstractmethod
    def evaluate_temporal(
        self,
        state: "TrajectoryState",
        ctx: "OptimizationContext",
    ) -> Tensor:
        """
        Compute temporal cost.

        Args:
            state: Trajectory state
            ctx: Optimization context

        Returns:
            Per-trajectory costs. Shape: (B,) or (B*K,)
        """
        pass

    def evaluate(
        self,
        state: "TrajectoryState",
        ctx: "OptimizationContext",
    ) -> Tensor:
        """Evaluate temporal cost."""
        # Flatten perturbations for computation
        if state._has_perturbations:
            flat_state = state.flatten_perturbations()
        else:
            flat_state = state

        return self.evaluate_temporal(flat_state, ctx)
