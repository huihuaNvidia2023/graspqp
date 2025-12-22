"""
Temporal costs - smoothness, velocity, acceleration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
from torch import Tensor

from .base import TemporalCost

if TYPE_CHECKING:
    from ..context import OptimizationContext
    from ..state import TrajectoryState


class VelocitySmoothnessCost(TemporalCost):
    """
    Cost to encourage smooth velocities.

    Penalizes large velocities (finite differences between frames).

    Config options:
        component: "hand", "object", or "both" (default: "both")
        hand_weight: Weight for hand velocity (default: 1.0)
        object_weight: Weight for object velocity (default: 1.0)
    """

    def __init__(
        self,
        name: str = "velocity_smoothness",
        weight: float = 1.0,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, weight, enabled, config, window_size=2)

        config = config or {}
        self.component = config.get("component", "both")
        self.hand_weight = config.get("hand_weight", 1.0)
        self.object_weight = config.get("object_weight", 1.0)

    def evaluate_temporal(
        self,
        state: "TrajectoryState",
        ctx: "OptimizationContext",
    ) -> Tensor:
        """
        Compute velocity smoothness cost.

        Returns:
            Per-trajectory costs. Shape: (B,)
        """
        B = state.B
        cost = torch.zeros(B, device=state.device)

        if self.component in ["hand", "both"]:
            # Hand velocity: (B, T-1, D)
            hand_vel = state.velocities("hand")
            hand_vel_sq = (hand_vel**2).sum(dim=-1)  # (B, T-1)

            # Apply mask for valid transitions
            if state.valid_mask is not None:
                # Valid transition = both frames valid
                valid_trans = state.valid_mask[:, :-1] & state.valid_mask[:, 1:]
                hand_vel_sq = hand_vel_sq * valid_trans.float()

            cost = cost + self.hand_weight * hand_vel_sq.sum(dim=-1)

        if self.component in ["object", "both"]:
            # Object velocity: (B, T-1, D)
            obj_vel = state.velocities("object")
            obj_vel_sq = (obj_vel**2).sum(dim=-1)  # (B, T-1)

            if state.valid_mask is not None:
                valid_trans = state.valid_mask[:, :-1] & state.valid_mask[:, 1:]
                obj_vel_sq = obj_vel_sq * valid_trans.float()

            cost = cost + self.object_weight * obj_vel_sq.sum(dim=-1)

        return cost


class AccelerationCost(TemporalCost):
    """
    Cost to encourage smooth accelerations (bounded jerk).

    Penalizes large accelerations (second-order finite differences).

    Config options:
        component: "hand", "object", or "both" (default: "both")
        hand_weight: Weight for hand acceleration (default: 1.0)
        object_weight: Weight for object acceleration (default: 1.0)
    """

    def __init__(
        self,
        name: str = "acceleration",
        weight: float = 0.1,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, weight, enabled, config, window_size=3)

        config = config or {}
        self.component = config.get("component", "both")
        self.hand_weight = config.get("hand_weight", 1.0)
        self.object_weight = config.get("object_weight", 1.0)

    def evaluate_temporal(
        self,
        state: "TrajectoryState",
        ctx: "OptimizationContext",
    ) -> Tensor:
        """
        Compute acceleration cost.

        Returns:
            Per-trajectory costs. Shape: (B,)
        """
        B = state.B

        # Need at least 3 frames for acceleration
        if state.T < 3:
            return torch.zeros(B, device=state.device)

        cost = torch.zeros(B, device=state.device)

        if self.component in ["hand", "both"]:
            # Hand acceleration: (B, T-2, D)
            hand_acc = state.accelerations("hand")
            hand_acc_sq = (hand_acc**2).sum(dim=-1)  # (B, T-2)

            # Apply mask
            if state.valid_mask is not None:
                # Valid acceleration = three consecutive frames valid
                valid_acc = state.valid_mask[:, :-2] & state.valid_mask[:, 1:-1] & state.valid_mask[:, 2:]
                hand_acc_sq = hand_acc_sq * valid_acc.float()

            cost = cost + self.hand_weight * hand_acc_sq.sum(dim=-1)

        if self.component in ["object", "both"]:
            obj_acc = state.accelerations("object")
            obj_acc_sq = (obj_acc**2).sum(dim=-1)

            if state.valid_mask is not None:
                valid_acc = state.valid_mask[:, :-2] & state.valid_mask[:, 1:-1] & state.valid_mask[:, 2:]
                obj_acc_sq = obj_acc_sq * valid_acc.float()

            cost = cost + self.object_weight * obj_acc_sq.sum(dim=-1)

        return cost


class JerkCost(TemporalCost):
    """
    Cost to encourage smooth jerk (third derivative).

    Penalizes large jerk for very smooth motion.
    """

    def __init__(
        self,
        name: str = "jerk",
        weight: float = 0.01,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, weight, enabled, config, window_size=4)

        config = config or {}
        self.component = config.get("component", "hand")

    def evaluate_temporal(
        self,
        state: "TrajectoryState",
        ctx: "OptimizationContext",
    ) -> Tensor:
        """
        Compute jerk cost.

        Returns:
            Per-trajectory costs. Shape: (B,)
        """
        B = state.B

        # Need at least 4 frames for jerk
        if state.T < 4:
            return torch.zeros(B, device=state.device)

        # Jerk = d(acceleration)/dt
        acc = state.accelerations(self.component)  # (B, T-2, D)
        jerk = (acc[:, 1:] - acc[:, :-1]) / state.dt  # (B, T-3, D)

        jerk_sq = (jerk**2).sum(dim=-1)  # (B, T-3)

        # Apply mask
        if state.valid_mask is not None:
            valid_jerk = (
                state.valid_mask[:, :-3]
                & state.valid_mask[:, 1:-2]
                & state.valid_mask[:, 2:-1]
                & state.valid_mask[:, 3:]
            )
            jerk_sq = jerk_sq * valid_jerk.float()

        return jerk_sq.sum(dim=-1)
