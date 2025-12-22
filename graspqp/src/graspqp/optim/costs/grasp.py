"""
Grasp-specific cost functions that wrap existing energy calculations.

These costs are designed for single-frame grasp optimization (T=1)
and provide compatibility with the existing graspqp energy functions.
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


class ContactDistanceCost(PerFrameCost):
    """
    Cost to ensure contact points touch the object surface.

    Equivalent to E_dis in fit.py.
    Penalizes distance between contact points and object surface.
    """

    def __init__(
        self,
        name: str = "contact_distance",
        weight: float = 100.0,
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
        Compute per-frame contact distance cost.

        Returns:
            Per-frame costs. Shape: (B, T)
        """
        B, T, D = state.hand_states.shape
        device = state.device

        # Flatten to (B*T, D) for batched FK
        flat_hand = state.flat_hand

        # Set hand parameters and get contact points
        ctx.hand_model.set_parameters(flat_hand)
        contact_points = ctx.hand_model.contact_points  # (B*T, n_contacts, 3)

        # Compute SDF at contact points
        distance, _ = ctx.object_model.cal_distance(contact_points)  # (B*T, n_contacts)

        # Cost is squared distance
        cost = (distance**2).sum(dim=-1)  # (B*T,)

        # Reshape to (B, T)
        return cost.reshape(B, T)


class ForceClosureCost(PerFrameCost):
    """
    Cost for grasp force closure / stability.

    Equivalent to E_fc in fit.py.
    Uses the existing GraspSpanMetric for QP-based force closure.
    """

    def __init__(
        self,
        name: str = "force_closure",
        weight: float = 1.0,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
        aggregation: str = "sum",
    ):
        super().__init__(name, weight, enabled, config, aggregation)

        config = config or {}
        self.svd_gain = config.get("svd_gain", 0.1)
        self._energy_fnc = None  # Set externally

    def set_energy_function(self, energy_fnc):
        """Set the grasp span metric function."""
        self._energy_fnc = energy_fnc

    def evaluate_frames(
        self,
        state: "TrajectoryState",
        ctx: "OptimizationContext",
    ) -> Tensor:
        """
        Compute per-frame force closure cost.

        Returns:
            Per-frame costs. Shape: (B, T)
        """
        B, T, D = state.hand_states.shape
        device = state.device

        if self._energy_fnc is None:
            return torch.zeros(B, T, device=device)

        # Flatten to (B*T, D)
        flat_hand = state.flat_hand

        # Set hand parameters
        ctx.hand_model.set_parameters(flat_hand)
        contact_points = ctx.hand_model.contact_points

        # Get contact normals from SDF
        distance, contact_normal = ctx.object_model.cal_distance(contact_points)

        # Compute force closure energy
        E_fc, _ = self._energy_fnc(
            contact_pts=contact_points,
            contact_normals=contact_normal,
            sdf=distance,
            cog=ctx.object_model.cog,
            with_solution=True,
            svd_gain=self.svd_gain,
        )

        # Reshape to (B, T)
        return E_fc.reshape(B, T)


class JointLimitCost(PerFrameCost):
    """
    Cost to keep joints within limits.

    Equivalent to E_joints in fit.py.
    """

    def __init__(
        self,
        name: str = "joint_limits",
        weight: float = 1.0,
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
        Compute per-frame joint limit cost.

        Returns:
            Per-frame costs. Shape: (B, T)
        """
        B, T, D = state.hand_states.shape
        device = state.device

        # Flatten to (B*T, D)
        flat_hand = state.flat_hand

        # Set hand parameters
        ctx.hand_model.set_parameters(flat_hand)

        # Get joint limit violation from hand model
        if hasattr(ctx.hand_model, "get_joint_limits_violations"):
            violations = ctx.hand_model.get_joint_limits_violations()  # (B*T,)
        else:
            violations = torch.zeros(B * T, device=device)

        # Reshape to (B, T)
        return violations.reshape(B, T)


class PriorPoseCost(PerFrameCost):
    """
    Cost to stay close to a prior hand pose.

    Equivalent to E_prior in fit.py.
    """

    def __init__(
        self,
        name: str = "prior_pose",
        weight: float = 10.0,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
        aggregation: str = "sum",
    ):
        super().__init__(name, weight, enabled, config, aggregation)
        self._prior_pose: Optional[Tensor] = None

    def set_prior_pose(self, prior_pose: Tensor):
        """Set the prior pose tensor."""
        self._prior_pose = prior_pose

    def evaluate_frames(
        self,
        state: "TrajectoryState",
        ctx: "OptimizationContext",
    ) -> Tensor:
        """
        Compute per-frame prior pose cost.

        Returns:
            Per-frame costs. Shape: (B, T)
        """
        B, T, D = state.hand_states.shape
        device = state.device

        if self._prior_pose is None:
            return torch.zeros(B, T, device=device)

        # Flatten to (B*T, D)
        flat_hand = state.flat_hand

        # Compute L2 distance to prior
        # Prior shape should be (B*T, D) or (B, D) for broadcasting
        if self._prior_pose.dim() == 2 and self._prior_pose.shape[0] == B:
            # (B, D) -> expand to (B*T, D) by repeating for each frame
            prior = self._prior_pose.unsqueeze(1).expand(B, T, D).reshape(B * T, D)
        else:
            prior = self._prior_pose

        cost = ((flat_hand - prior) ** 2).sum(dim=-1)  # (B*T,)

        # Reshape to (B, T)
        return cost.reshape(B, T)
