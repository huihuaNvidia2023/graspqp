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

        # Get SDF distance and normals at contact points (CACHED - expensive operation!)
        distance, contact_normal = ctx.get_contact_sdf_cached(flat_hand)  # (B*T, n_contacts)

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

    OPTIMIZATION: Only computes QP when contacts are established (mean contact
    distance below threshold). This avoids wasteful QP computations and
    numerical instability warnings when fingers aren't touching the object.

    Config options:
        svd_gain: SVD regularization gain (default: 0.1)
        contact_threshold: Mean contact distance threshold for QP computation (default: 0.01)
                          Set to None or inf to always compute QP.
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
        self.contact_threshold = config.get("contact_threshold", 0.01)
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

        Only computes QP for samples where mean contact distance is below
        contact_threshold. Returns zero cost for samples without established contacts.

        Returns:
            Per-frame costs. Shape: (B, T)
        """
        B, T, D = state.hand_states.shape
        N = B * T  # Flattened batch size
        device = state.device

        if self._energy_fnc is None:
            return torch.zeros(B, T, device=device)

        # Flatten to (B*T, D)
        flat_hand = state.flat_hand

        # Get contact points using cached FK
        contact_points = ctx.get_contact_points_cached(flat_hand)

        # Get SDF distance and normals (CACHED - expensive operation!)
        distance, contact_normal = ctx.get_contact_sdf_cached(flat_hand)

        # Check which samples have established contacts
        # Mean absolute distance across all contact points per sample
        mean_distance = distance.abs().mean(dim=-1)  # (N,)

        # Determine which samples should compute QP
        if self.contact_threshold is not None and self.contact_threshold < float('inf'):
            in_contact_mask = mean_distance < self.contact_threshold  # (N,)
            n_in_contact = in_contact_mask.sum().item()

            # If no samples are in contact, return zeros (skip expensive QP)
            if n_in_contact == 0:
                return torch.zeros(B, T, device=device)

            # If all samples are in contact, compute normally
            if n_in_contact == N:
                in_contact_mask = None  # Flag to compute all
        else:
            in_contact_mask = None  # Compute all (no threshold)

        # Compute force closure energy (QP solver - expensive!)
        with ctx._profile_section("qp_solver"):
            if in_contact_mask is None:
                # Compute for all samples
                E_fc, _ = self._energy_fnc(
                    contact_pts=contact_points,
                    contact_normals=contact_normal,
                    sdf=distance,
                    cog=ctx.object_model.cog,
                    with_solution=True,
                    svd_gain=self.svd_gain,
                )
            else:
                # Compute only for samples with established contacts
                E_fc = torch.zeros(N, device=device)

                # Extract in-contact samples
                contact_pts_subset = contact_points[in_contact_mask]
                contact_normals_subset = contact_normal[in_contact_mask]
                distance_subset = distance[in_contact_mask]

                # COG needs to match batch size
                cog = ctx.object_model.cog
                if cog.shape[0] == N:
                    cog_subset = cog[in_contact_mask]
                else:
                    # COG is shared across batches
                    cog_subset = cog

                E_fc_subset, _ = self._energy_fnc(
                    contact_pts=contact_pts_subset,
                    contact_normals=contact_normals_subset,
                    sdf=distance_subset,
                    cog=cog_subset,
                    with_solution=True,
                    svd_gain=self.svd_gain,
                )
                E_fc[in_contact_mask] = E_fc_subset

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

        if self._prior_pose is None:
            return torch.zeros(B, T, device=state.device)

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
