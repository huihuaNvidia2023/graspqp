"""
Reference tracking cost - stay close to the video trajectory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
from torch import Tensor

from .base import PerFrameCost

if TYPE_CHECKING:
    from ..context import OptimizationContext
    from ..state import TrajectoryState


class ReferenceTrackingCost(PerFrameCost):
    """
    Cost to stay close to the original video trajectory.

    Computes weighted L2 distance between current state and reference.

    Config options:
        hand_weight: Weight for hand pose error (default: 1.0)
        object_weight: Weight for object pose error (default: 1.0)
        hand_position_weight: Weight for wrist position (default: 1.0)
        hand_rotation_weight: Weight for wrist rotation (default: 1.0)
        finger_weight: Weight for finger joints (default: 1.0)
        use_confidence: Whether to weight by per-frame confidence (default: False)
    """

    def __init__(
        self,
        name: str = "reference_tracking",
        weight: float = 100.0,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
        aggregation: str = "sum",
    ):
        super().__init__(name, weight, enabled, config, aggregation)

        # Parse config
        config = config or {}
        self.hand_weight = config.get("hand_weight", 1.0)
        self.object_weight = config.get("object_weight", 1.0)
        self.hand_position_weight = config.get("hand_position_weight", 1.0)
        self.hand_rotation_weight = config.get("hand_rotation_weight", 1.0)
        self.finger_weight = config.get("finger_weight", 1.0)
        self.use_confidence = config.get("use_confidence", False)

        # Hand state slicing (depends on hand_type)
        # Default: [pos(3), rot6d(6), joints(N)]
        self.pos_slice = slice(0, 3)
        self.rot_slice = slice(3, 9)
        self.joint_slice = slice(9, None)

    @property
    def provided_cache_keys(self):
        return ["reference_hand_error", "reference_object_error"]

    def evaluate_frames(
        self,
        state: "TrajectoryState",
        ctx: "OptimizationContext",
    ) -> Tensor:
        """
        Compute per-frame reference tracking error.

        Returns:
            Per-frame costs. Shape: (B, T) or (B*K, T)
        """
        reference = ctx.reference
        if reference is None:
            return torch.zeros(state.B, state.T, device=state.device)

        B, T = state.B, state.T

        # Expand reference if state has flattened perturbations
        # state: (B*K, T, D), reference: (B, T, D)
        # We need to expand reference to match
        ref_hand = reference.hand_states
        ref_object = reference.object_states

        if state.B != reference.B:
            # State was flattened from (B, K, T, D) to (B*K, T, D)
            K = state.B // reference.B
            ref_hand = ref_hand.unsqueeze(1).expand(-1, K, -1, -1).reshape(state.B, T, -1)
            ref_object = ref_object.unsqueeze(1).expand(-1, K, -1, -1).reshape(state.B, T, -1)

        # Compute hand error with component weights
        hand_error = torch.zeros(B, T, device=state.device)

        # Position error
        pos_error = (state.hand_states[..., self.pos_slice] - ref_hand[..., self.pos_slice]) ** 2
        hand_error = hand_error + self.hand_position_weight * pos_error.sum(dim=-1)

        # Rotation error
        rot_error = (state.hand_states[..., self.rot_slice] - ref_hand[..., self.rot_slice]) ** 2
        hand_error = hand_error + self.hand_rotation_weight * rot_error.sum(dim=-1)

        # Joint error
        joint_error = (state.hand_states[..., self.joint_slice] - ref_hand[..., self.joint_slice]) ** 2
        hand_error = hand_error + self.finger_weight * joint_error.sum(dim=-1)

        # Object error
        object_error = (state.object_states - ref_object) ** 2
        object_error = object_error.sum(dim=-1)  # (B, T)

        # Combine
        total_error = self.hand_weight * hand_error + self.object_weight * object_error

        # Apply confidence weighting if available
        if self.use_confidence and reference.confidence is not None:
            confidence = reference.confidence
            if state.B != reference.B:
                K = state.B // reference.B
                confidence = confidence.unsqueeze(1).expand(-1, K, -1).reshape(state.B, T)
            total_error = total_error * confidence

        # Cache for debugging
        ctx.set_cached("reference_hand_error", hand_error.detach(), scope="step")
        ctx.set_cached("reference_object_error", object_error.detach(), scope="step")

        return total_error
