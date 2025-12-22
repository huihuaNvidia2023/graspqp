"""
Penetration cost - penalize hand-object interpenetration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import PerFrameCost

if TYPE_CHECKING:
    from ..context import OptimizationContext
    from ..state import TrajectoryState


class PenetrationCost(PerFrameCost):
    """
    Cost to prevent hand-object penetration.

    Computes SDF values for hand surface points and penalizes negative values
    (indicating penetration into the object).

    Config options:
        use_capsules: Use capsule approximation for speed (default: False)
        n_surface_points: Number of surface points to sample (default: None, use all)
    """

    def __init__(
        self,
        name: str = "penetration",
        weight: float = 1000.0,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
        aggregation: str = "sum",
    ):
        super().__init__(name, weight, enabled, config, aggregation)

        config = config or {}
        self.use_capsules = config.get("use_capsules", False)
        self.n_surface_points = config.get("n_surface_points", None)

    @property
    def provided_cache_keys(self):
        return ["hand_sdf", "penetration_depth"]

    def evaluate_frames(
        self,
        state: "TrajectoryState",
        ctx: "OptimizationContext",
    ) -> Tensor:
        """
        Compute per-frame penetration cost.

        Returns:
            Per-frame costs. Shape: (B, T)
        """
        B, T, D = state.hand_states.shape
        device = state.device

        # Flatten to (B*T, D) for batched FK
        flat_hand = state.flat_hand  # (B*T, D)

        # Get surface points from hand model
        ctx.hand_model.set_parameters(flat_hand)

        if self.use_capsules:
            # Use capsule/sphere approximation (faster)
            # TODO: Implement capsule proxy
            surface_points = ctx.hand_model.get_surface_points()
        else:
            # Use full surface points
            surface_points = ctx.hand_model.get_surface_points()  # (B*T, n_pts, 3)

        if self.n_surface_points is not None and surface_points.shape[1] > self.n_surface_points:
            # Subsample for efficiency
            idx = torch.randperm(surface_points.shape[1], device=device)[: self.n_surface_points]
            surface_points = surface_points[:, idx]

        # Compute SDF values
        # Note: Since we're using Hand_T_Object coordinates, points are already in object frame
        sdf_values, _ = ctx.object_model.cal_distance(surface_points)  # (B*T, n_pts)

        # Penetration = max(0, -sdf) (negative sdf means inside object)
        penetration = F.relu(-sdf_values)  # (B*T, n_pts)

        # Sum over points
        per_config = penetration.sum(dim=-1)  # (B*T,)

        # Reshape to (B, T)
        per_frame = per_config.reshape(B, T)

        # Cache for debugging
        ctx.set_cached("hand_sdf", sdf_values.detach(), scope="step")
        ctx.set_cached("penetration_depth", penetration.detach(), scope="step")

        return per_frame


class SelfPenetrationCost(PerFrameCost):
    """
    Cost to prevent finger-finger self-collision.

    Uses the hand model's self-penetration computation.
    """

    def __init__(
        self,
        name: str = "self_penetration",
        weight: float = 10.0,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
        aggregation: str = "sum",
    ):
        super().__init__(name, weight, enabled, config, aggregation)

    def evaluate_frames(
        self,
        state: "TrajectoryState",
        ctx: "OptimizationContext",
    ) -> Tensor:
        """
        Compute per-frame self-penetration cost.

        Returns:
            Per-frame costs. Shape: (B, T)
        """
        B, T, D = state.hand_states.shape

        # Flatten to (B*T, D) for batched computation
        flat_hand = state.flat_hand

        # Set hand parameters
        ctx.hand_model.set_parameters(flat_hand)

        # Compute self-penetration using hand model
        if hasattr(ctx.hand_model, "self_penetration"):
            self_pen = ctx.hand_model.self_penetration()  # (B*T,)
        else:
            # Fallback: return zeros if not supported
            self_pen = torch.zeros(B * T, device=state.device)

        # Reshape to (B, T)
        per_frame = self_pen.reshape(B, T)

        return per_frame
