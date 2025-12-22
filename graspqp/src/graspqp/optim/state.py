"""
Trajectory state representations for optimization.

Coordinate System Convention:
- hand_states: Hand_T_Object (pose of hand relative to object frame)
- object_states: Object_T_World (pose of object in world frame)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class TrajectoryState:
    """
    Optimization variables for trajectory optimization.

    Supports both single perturbation (B, T, D) and multi-perturbation (B, K, T, D) layouts.

    Attributes:
        hand_states: Hand poses relative to object. Shape: (B, T, D_hand) or (B, K, T, D_hand)
        object_states: Object poses in world frame. Shape: (B, T, D_obj) or (B, K, T, D_obj)
        valid_mask: Mask for valid frames (1=valid, 0=padding). Shape: (B, T) or (B, K, T)
        dt: Timestep in seconds (default 0.1s)
    """

    hand_states: Tensor
    object_states: Tensor
    valid_mask: Optional[Tensor] = None
    dt: float = 0.1

    # Derived attributes (set in __post_init__)
    # The `field(init=False)` keyword is used here to indicate that these attributes
    # (B, K, T) are NOT expected as constructor arguments, but will instead be set
    # later—specifically, in the __post_init__ method based on the shape of input tensors.
    B: int = field(init=False)
    K: int = field(init=False)
    T: int = field(init=False)
    D_hand: int = field(init=False)
    D_obj: int = field(init=False)
    _has_perturbations: bool = field(init=False)

    def __post_init__(self):
        # Detect layout from shape
        if self.hand_states.dim() == 3:
            self.B, self.T, self.D_hand = self.hand_states.shape
            self.K = 1
            self._has_perturbations = False
        elif self.hand_states.dim() == 4:
            self.B, self.K, self.T, self.D_hand = self.hand_states.shape
            self._has_perturbations = True
        else:
            raise ValueError(f"hand_states must be 3D or 4D, got {self.hand_states.dim()}D")

        self.D_obj = self.object_states.shape[-1]

        # Create default valid_mask if not provided
        if self.valid_mask is None:
            if self._has_perturbations:
                self.valid_mask = torch.ones(self.B, self.K, self.T, device=self.hand_states.device, dtype=torch.bool)
            else:
                self.valid_mask = torch.ones(self.B, self.T, device=self.hand_states.device, dtype=torch.bool)

    @property
    def device(self) -> torch.device:
        return self.hand_states.device

    @property
    def flat_hand(self) -> Tensor:
        """Flatten to (B*K*T, D_hand) for batched FK operations."""
        if self._has_perturbations:
            return self.hand_states.reshape(self.B * self.K * self.T, self.D_hand)
        return self.hand_states.reshape(self.B * self.T, self.D_hand)

    @property
    def flat_object(self) -> Tensor:
        """Flatten to (B*K*T, D_obj) for batched operations."""
        if self._has_perturbations:
            return self.object_states.reshape(self.B * self.K * self.T, self.D_obj)
        return self.object_states.reshape(self.B * self.T, self.D_obj)

    @property
    def flat_mask(self) -> Tensor:
        """Flatten valid_mask to (B*K*T,) or (B*T,)."""
        if self._has_perturbations:
            return self.valid_mask.reshape(self.B * self.K * self.T)
        return self.valid_mask.reshape(self.B * self.T)

    @property
    def lengths(self) -> Tensor:
        """Actual length of each trajectory. Shape: (B,) or (B, K)."""
        return self.valid_mask.sum(dim=-1)

    def get_frame(self, t: int) -> Tuple[Tensor, Tensor]:
        """
        Get hand and object states at frame t.

        Returns:
            hand: (B, D_hand) or (B, K, D_hand)
            obj: (B, D_obj) or (B, K, D_obj)
        """
        if self._has_perturbations:
            return self.hand_states[:, :, t], self.object_states[:, :, t]
        return self.hand_states[:, t], self.object_states[:, t]

    def velocities(self, component: str = "hand") -> Tensor:
        """
        Compute velocities via finite difference.

        Args:
            component: "hand" or "object"

        Returns:
            Velocities with shape (B, T-1, D) or (B, K, T-1, D)
        """
        states = self.hand_states if component == "hand" else self.object_states
        if self._has_perturbations:
            vel = (states[:, :, 1:] - states[:, :, :-1]) / self.dt
        else:
            vel = (states[:, 1:] - states[:, :-1]) / self.dt
        return vel

    def accelerations(self, component: str = "hand") -> Tensor:
        """
        Compute accelerations via second-order finite difference.

        Returns:
            Accelerations with shape (B, T-2, D) or (B, K, T-2, D)
        """
        vel = self.velocities(component)
        if self._has_perturbations:
            acc = (vel[:, :, 1:] - vel[:, :, :-1]) / self.dt
        else:
            acc = (vel[:, 1:] - vel[:, :-1]) / self.dt
        return acc

    def flatten_perturbations(self) -> "TrajectoryState":
        """
        Flatten (B, K, T, D) to (B*K, T, D) for computation.
        Returns self if already flat.
        """
        if not self._has_perturbations:
            return self

        return TrajectoryState(
            hand_states=self.hand_states.reshape(self.B * self.K, self.T, self.D_hand),
            object_states=self.object_states.reshape(self.B * self.K, self.T, self.D_obj),
            valid_mask=self.valid_mask.reshape(self.B * self.K, self.T),
            dt=self.dt,
        )

    def unflatten_perturbations(self, B: int, K: int) -> "TrajectoryState":
        """Reshape (B*K, T, D) back to (B, K, T, D)."""
        return TrajectoryState(
            hand_states=self.hand_states.reshape(B, K, self.T, self.D_hand),
            object_states=self.object_states.reshape(B, K, self.T, self.D_obj),
            valid_mask=self.valid_mask.reshape(B, K, self.T),
            dt=self.dt,
        )

    def clone(self) -> "TrajectoryState":
        """Create a deep copy."""
        return TrajectoryState(
            hand_states=self.hand_states.clone(),
            object_states=self.object_states.clone(),
            valid_mask=self.valid_mask.clone() if self.valid_mask is not None else None,
            dt=self.dt,
        )

    def detach(self) -> "TrajectoryState":
        """Detach from computation graph."""
        return TrajectoryState(
            hand_states=self.hand_states.detach(),
            object_states=self.object_states.detach(),
            valid_mask=self.valid_mask.detach() if self.valid_mask is not None else None,
            dt=self.dt,
        )

    def requires_grad_(self, requires_grad: bool = True) -> "TrajectoryState":
        """Set requires_grad for optimization."""
        self.hand_states.requires_grad_(requires_grad)
        self.object_states.requires_grad_(requires_grad)
        return self

    def to(self, device: torch.device) -> "TrajectoryState":
        """Move to device."""
        return TrajectoryState(
            hand_states=self.hand_states.to(device),
            object_states=self.object_states.to(device),
            valid_mask=self.valid_mask.to(device) if self.valid_mask is not None else None,
            dt=self.dt,
        )

    def to_legacy_dict(self, frame: int = -1) -> Dict[str, Any]:
        """
        Export to legacy format compatible with visualize_result.py.

        Args:
            frame: Which frame to export (-1 for last frame)
        """
        # Get single frame
        if self._has_perturbations:
            # Use first perturbation for legacy format
            hand = self.hand_states[:, 0, frame]  # (B, D_hand)
            obj = self.object_states[:, 0, frame]  # (B, D_obj)
        else:
            hand = self.hand_states[:, frame]
            obj = self.object_states[:, frame]

        # Convert hand state to legacy parameter format
        # Assuming: [root_pos(3), root_rot6d(6), joints(N)]
        parameters = {
            "root_pose": torch.cat([hand[:, :3], hand[:, 3:7]], dim=-1),  # Placeholder
        }

        return {
            "parameters": parameters,
            "object_pose": obj,
        }

    def to_trajectory_dict(self) -> Dict[str, Any]:
        """Export full trajectory data."""
        return {
            "trajectory": {
                "n_frames": self.T,
                "dt": self.dt,
                "hand_states": self.hand_states.detach().cpu(),
                "object_states": self.object_states.detach().cpu(),
            },
            "valid_mask": self.valid_mask.detach().cpu() if self.valid_mask is not None else None,
        }

    @staticmethod
    def from_reference(
        reference: "ReferenceTrajectory",
        n_perturbations: int = 1,
        perturbation_scale: float = 0.01,
    ) -> "TrajectoryState":
        """
        Create initial state with K perturbations from reference.

        Args:
            reference: The reference trajectory from video
            n_perturbations: Number of perturbed copies (K)
            perturbation_scale: Standard deviation of Gaussian noise
        """
        B, T, D_hand = reference.hand_states.shape
        D_obj = reference.object_states.shape[-1]
        device = reference.hand_states.device
        K = n_perturbations

        if K == 1:
            return TrajectoryState(
                hand_states=reference.hand_states.clone(),
                object_states=reference.object_states.clone(),
                valid_mask=reference.valid_mask.clone() if reference.valid_mask is not None else None,
                dt=reference.dt,
            )

        # Generate K perturbations
        hand_noise = torch.randn(B, K, T, D_hand, device=device) * perturbation_scale
        obj_noise = torch.randn(B, K, T, D_obj, device=device) * perturbation_scale

        hand_states = reference.hand_states.unsqueeze(1) + hand_noise  # (B, K, T, D)
        object_states = reference.object_states.unsqueeze(1) + obj_noise

        valid_mask = None
        if reference.valid_mask is not None:
            valid_mask = reference.valid_mask.unsqueeze(1).expand(B, K, T).clone()

        return TrajectoryState(
            hand_states=hand_states,
            object_states=object_states,
            valid_mask=valid_mask,
            dt=reference.dt,
        )


@dataclass
class ReferenceTrajectory:
    """
    The original video trajectory (fixed, not optimized).

    This is the TARGET we want to stay close to while satisfying physical constraints.

    Contact Model:
        - contact_fingers: HIGH-LEVEL info specifying which FINGERS are in contact
        - The optimizer determines which specific CONTACT POINTS on those fingers are used
        - n_contacts: Minimum number of contact points required

    Attributes:
        hand_states: Hand poses relative to object from video. Shape: (B, T, D_hand)
        object_states: Object poses from video. Shape: (B, T, D_obj)
        contact_fingers: List of finger/link names that are in contact (e.g., ["thumb", "index", "middle"])
        n_contacts: Minimum number of contact points required per grasp
        valid_mask: Mask for valid frames. Shape: (B, T)
        confidence: Optional per-frame confidence. Shape: (B, T)
        dt: Timestep in seconds
    """

    hand_states: Tensor
    object_states: Tensor
    contact_fingers: Optional[List[str]] = None  # Which fingers are in contact (high-level)
    n_contacts: int = 8  # Minimum contact points required
    valid_mask: Optional[Tensor] = None
    confidence: Optional[Tensor] = None
    dt: float = 0.1

    # Metadata
    hand_type: str = "allegro"

    def __post_init__(self):
        self.B, self.T, self.D_hand = self.hand_states.shape
        self.D_obj = self.object_states.shape[-1]

        if self.valid_mask is None:
            self.valid_mask = torch.ones(self.B, self.T, device=self.hand_states.device, dtype=torch.bool)

    @property
    def device(self) -> torch.device:
        return self.hand_states.device

    def to(self, device: torch.device) -> "ReferenceTrajectory":
        """Move to device."""
        return ReferenceTrajectory(
            hand_states=self.hand_states.to(device),
            object_states=self.object_states.to(device),
            contact_fingers=self.contact_fingers,
            n_contacts=self.n_contacts,
            valid_mask=self.valid_mask.to(device) if self.valid_mask is not None else None,
            confidence=self.confidence.to(device) if self.confidence is not None else None,
            dt=self.dt,
            hand_type=self.hand_type,
        )


class ResultSelector:
    """Select best results from multi-perturbation optimization."""

    @staticmethod
    def select_best_valid(
        state: TrajectoryState,
        energies: Tensor,
        valid: Tensor,
    ) -> Tuple[TrajectoryState, Tensor, Tensor]:
        """
        Select best valid trajectory per batch.

        Args:
            state: Optimized state with shape (B, K, T, D)
            energies: Total energy per trajectory. Shape: (B, K)
            valid: Whether each trajectory satisfies constraints. Shape: (B, K)

        Returns:
            best_state: Best valid trajectory per batch. Shape: (B, T, D)
            best_energy: Energy of selected trajectory. Shape: (B,)
            success: Whether any valid solution was found. Shape: (B,)
        """
        assert state._has_perturbations, "State must have perturbations (B, K, T, D)"

        B, K = energies.shape
        device = energies.device

        # Mask invalid with large energy
        masked_energies = energies.clone()
        masked_energies[~valid] = float("inf")

        # Find best per batch
        best_k = masked_energies.argmin(dim=1)  # (B,)
        success = valid.any(dim=1)  # (B,)

        # Gather best trajectories
        batch_idx = torch.arange(B, device=device)
        best_hand = state.hand_states[batch_idx, best_k]  # (B, T, D)
        best_object = state.object_states[batch_idx, best_k]  # (B, T, D)
        best_mask = state.valid_mask[batch_idx, best_k] if state.valid_mask is not None else None
        best_energy = energies[batch_idx, best_k]  # (B,)

        best_state = TrajectoryState(
            hand_states=best_hand,
            object_states=best_object,
            valid_mask=best_mask,
            dt=state.dt,
        )

        return best_state, best_energy, success

    @staticmethod
    def select_all_valid(
        state: TrajectoryState,
        energies: Tensor,
        valid: Tensor,
    ) -> Tuple[List[TrajectoryState], List[Tensor]]:
        """
        Return all valid trajectories per batch.

        Returns:
            List of TrajectoryState per batch (variable length)
            List of energies per batch
        """
        B, K = energies.shape

        all_states = []
        all_energies = []

        for b in range(B):
            valid_k = valid[b].nonzero(as_tuple=True)[0]
            if len(valid_k) > 0:
                states_b = TrajectoryState(
                    hand_states=state.hand_states[b, valid_k],  # (n_valid, T, D)
                    object_states=state.object_states[b, valid_k],
                    valid_mask=state.valid_mask[b, valid_k] if state.valid_mask is not None else None,
                    dt=state.dt,
                )
                all_states.append(states_b)
                all_energies.append(energies[b, valid_k])
            else:
                all_states.append(None)
                all_energies.append(None)

        return all_states, all_energies
