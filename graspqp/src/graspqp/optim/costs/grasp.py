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

    Matches fit.py's E_dis with gendexgrasp method:
    E_dis = ((1 - sum((-vC) * nH)).exp() * distance.abs()).sum(-1)

    Where:
    - vC = contact_normal from object SDF (pointing outward from object)
    - nH = hand_model.contact_normals (pointing outward from hand)
    - The dot product measures alignment between hand and object normals

    Config options:
        method: "gendexgrasp" (default) or "dexgraspnet"
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
        config = config or {}
        self.method = config.get("method", "gendexgrasp")

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

        # Flatten to (B*T, D) for batched FK
        flat_hand = state.flat_hand

        # Get contact points using cached FK
        contact_points = ctx.get_contact_points_cached(flat_hand)  # (B*T, n_contacts, 3)

        # Compute SDF and normals at contact points
        distance, contact_normal = ctx.object_model.cal_distance(contact_points)  # (B*T, n_contacts)

        if self.method == "dexgraspnet":
            # Simple method: sum of absolute distances
            cost = distance.abs().sum(dim=-1)  # (B*T,)
        else:
            # gendexgrasp method (default, matches fit.py)
            # vC = object normal (pointing outward from object)
            # nH = hand contact normals (pointing outward from hand)
            vC = contact_normal  # (B*T, n_contacts, 3)
            nH = ctx.hand_model.contact_normals  # (B*T, n_contacts, 3)

            # Dot product of -vC and nH: measures how aligned the normals are
            # When contact is good, -vC (into object) aligns with nH (out of hand)
            dot_product = torch.sum((-vC) * nH, dim=-1)  # (B*T, n_contacts)

            # Cost: exp(1 - dot_product) * |distance|
            # - When aligned (dot=1): exp(0) * |d| = |d|
            # - When misaligned (dot=-1): exp(2) * |d| ≈ 7.4 * |d|
            cost = ((1 - dot_product).exp() * distance.abs()).sum(dim=-1)  # (B*T,)

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

        # Get contact points using cached FK
        contact_points = ctx.get_contact_points_cached(flat_hand)

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

        # Ensure hand model is configured (uses cache)
        ctx.ensure_hand_configured(flat_hand)

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

    Matches fit.py's compute_prior_energy:
    - Translation: L2 distance
    - Rotation: Geodesic distance using 6D ortho representation
    - Joints: L2 distance with 0.1 weight
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

        Matches compute_prior_energy:
        - E_trans = (translation_diff ** 2).sum(-1)
        - E_rot = geodesic_distance(R_current, R_prior)
        - E_joints = (joint_diff ** 2).sum(-1) * 0.1

        Returns:
            Per-frame costs. Shape: (B, T)
        """
        from graspqp.utils.transforms import robust_compute_rotation_matrix_from_ortho6d

        B, T, D = state.hand_states.shape
        device = state.device

        if self._prior_pose is None:
            return torch.zeros(B, T, device=device)

        # Flatten to (B*T, D)
        flat_hand = state.flat_hand

        # Prior shape should be (B*T, D) or (B, D) for broadcasting
        if self._prior_pose.dim() == 2 and self._prior_pose.shape[0] == B:
            # (B, D) -> expand to (B*T, D) by repeating for each frame
            prior = self._prior_pose.unsqueeze(1).expand(B, T, D).reshape(B * T, D)
        else:
            prior = self._prior_pose

        # Translation deviation (first 3 dims)
        E_trans = ((flat_hand[:, :3] - prior[:, :3]) ** 2).sum(-1)

        # Rotation deviation (geodesic distance, dims 3:9)
        R_current = robust_compute_rotation_matrix_from_ortho6d(flat_hand[:, 3:9])
        R_prior = robust_compute_rotation_matrix_from_ortho6d(prior[:, 3:9])

        # Geodesic distance: arccos((trace(R1^T R2) - 1) / 2)
        R_diff = R_current.transpose(1, 2) @ R_prior
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        E_rot = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))

        # Joint deviation (dims 9:, with 0.1 weight as in compute_prior_energy)
        E_joints = ((flat_hand[:, 9:] - prior[:, 9:]) ** 2).sum(-1) * 0.1

        # Total (NOTE: compute_prior_energy multiplies by prior_weight internally,
        # but we handle weight in the base class, so just return the sum)
        cost = E_trans + E_rot + E_joints  # (B*T,)

        # Reshape to (B, T)
        return cost.reshape(B, T)
