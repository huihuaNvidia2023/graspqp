"""
Optimization problem definition.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import yaml
from torch import Tensor

from .context import OptimizationContext
from .costs.base import CostFunction
from .costs.registry import create_cost

if TYPE_CHECKING:
    from .state import TrajectoryState


class OptimizationProblem:
    """
    Defines an optimization problem as a collection of weighted costs.

    The total energy is the weighted sum of all enabled cost functions.

    Attributes:
        context: Shared optimization context
        costs: Ordered dict of cost functions
        profiler: Optional profiler for timing cost evaluations
    """

    def __init__(self, context: OptimizationContext, profiler=None):
        self.context = context
        self.costs: OrderedDict[str, CostFunction] = OrderedDict()
        self._weight_schedule: Dict[str, Any] = {}
        self.profiler = profiler

    def add_cost(self, cost: CostFunction):
        """Add a cost function to the problem."""
        if cost.name in self.costs:
            raise ValueError(f"Cost with name '{cost.name}' already exists")
        self.costs[cost.name] = cost

    def remove_cost(self, name: str):
        """Remove a cost function by name."""
        if name not in self.costs:
            raise ValueError(f"Cost '{name}' not found")
        del self.costs[name]

    def get_cost(self, name: str) -> CostFunction:
        """Get a cost function by name."""
        if name not in self.costs:
            raise ValueError(f"Cost '{name}' not found")
        return self.costs[name]

    def set_weight(self, name: str, weight: float):
        """Set the weight of a cost function."""
        self.costs[name].weight = weight

    def set_enabled(self, name: str, enabled: bool):
        """Enable or disable a cost function."""
        self.costs[name].enabled = enabled

    def _profile_section(self, name: str):
        """Context manager for profiling a section."""
        from contextlib import nullcontext

        if self.profiler is not None:
            return self.profiler.section(name)
        return nullcontext()

    def total_energy(self, state: "TrajectoryState") -> Tensor:
        """
        Compute total weighted energy.

        Args:
            state: Current trajectory state

        Returns:
            Total energy per trajectory. Shape: (B,) or (B*K,)
        """
        # Determine output shape
        if state._has_perturbations:
            B_out = state.B * state.K
        else:
            B_out = state.B

        total = torch.zeros(B_out, device=state.device)

        for cost in self.costs.values():
            if cost.enabled:
                with self._profile_section(f"costs.{cost.name}"):
                    energy = cost(state, self.context)
                total = total + energy

        return total

    def evaluate_all(self, state: "TrajectoryState") -> Dict[str, Tensor]:
        """
        Evaluate all costs and return per-cost energies.

        Args:
            state: Current trajectory state

        Returns:
            Dict mapping cost name to energy tensor
        """
        results = {}
        for name, cost in self.costs.items():
            if cost.enabled:
                results[name] = cost(state, self.context)
            else:
                # Return zeros for disabled costs
                if state._has_perturbations:
                    results[name] = torch.zeros(state.B * state.K, device=state.device)
                else:
                    results[name] = torch.zeros(state.B, device=state.device)
        return results

    def cost_breakdown(self, state: "TrajectoryState") -> Dict[str, float]:
        """
        Get mean cost values for logging.

        Returns:
            Dict mapping cost name to mean value
        """
        energies = self.evaluate_all(state)
        return {name: e.mean().item() for name, e in energies.items()}

    @classmethod
    def from_config(
        cls,
        config_path: str,
        context: OptimizationContext,
    ) -> "OptimizationProblem":
        """
        Create problem from YAML configuration file.

        Expected format:
        ```yaml
        problem:
          costs:
            - type: ReferenceTrackingCost
              name: reference_tracking
              weight: 100.0
              config:
                hand_weight: 1.0
            - type: PenetrationCost
              weight: 1000.0
            ...
        ```
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        problem = cls(context)

        # Load costs
        costs_config = config.get("problem", {}).get("costs", [])
        for cost_cfg in costs_config:
            cost = create_cost(cost_cfg)
            problem.add_cost(cost)

        return problem

    @classmethod
    def from_dict(
        cls,
        config: Dict[str, Any],
        context: OptimizationContext,
    ) -> "OptimizationProblem":
        """Create problem from configuration dict."""
        problem = cls(context)

        costs_config = config.get("costs", [])
        for cost_cfg in costs_config:
            cost = create_cost(cost_cfg)
            problem.add_cost(cost)

        return problem

    def __repr__(self) -> str:
        cost_strs = [f"  {name}: weight={c.weight}, enabled={c.enabled}" for name, c in self.costs.items()]
        return f"OptimizationProblem(\n" + "\n".join(cost_strs) + "\n)"
