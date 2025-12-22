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

    Matches fit.py's E_pen computation:
    - Get object's surface points
    - Compute hand model's SDF at those points
    - Penalize positive values (object points INSIDE hand)

    This checks if the object is penetrating INTO the hand.

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

        Matches fit.py's E_pen:
        - object_surface_points = object_model.surface_points_tensor * object_scale
        - distances = hand_model.cal_distance(object_surface_points)
        - distances[distances <= 0] = 0  # Interior is positive in hand SDF
        - E_pen = distances.sum(-1)

        Returns:
            Per-frame costs. Shape: (B, T)
        """
        B, T, D = state.hand_states.shape
        device = state.device

        # Ensure hand model is configured
        flat_hand = state.flat_hand  # (B*T, D)
        ctx.ensure_hand_configured(flat_hand)

        # Get OBJECT's surface points (not hand's!)
        # Same as fit.py: object_model.surface_points_tensor * object_scale
        object_scale = ctx.object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
        object_surface_points = ctx.object_model.surface_points_tensor * object_scale  # (B*T, n_samples, 3)

        # Compute HAND's SDF at object surface points
        # hand_model.cal_distance: interior is positive, exterior is negative
        distances = ctx.hand_model.cal_distance(object_surface_points)  # (B*T, n_samples)

        # Penalize positive values (object points inside hand)
        distances = distances.clone()
        distances[distances <= 0] = 0

        # Sum over points
        per_config = distances.sum(dim=-1)  # (B*T,)

        # Reshape to (B, T)
        per_frame = per_config.reshape(B, T)

        # Cache for debugging
        ctx.set_cached("hand_sdf", distances.detach(), scope="step")
        ctx.set_cached("penetration_depth", distances.detach(), scope="step")

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

        # Ensure hand model is configured (uses cache)
        ctx.ensure_hand_configured(flat_hand)

        # Compute self-penetration using hand model
        if hasattr(ctx.hand_model, "self_penetration"):
            self_pen = ctx.hand_model.self_penetration()  # (B*T,)
        else:
            # Fallback: return zeros if not supported
            self_pen = torch.zeros(B * T, device=state.device)

        # Reshape to (B, T)
        per_frame = self_pen.reshape(B, T)

        return per_frame
