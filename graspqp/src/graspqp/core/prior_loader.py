"""
Prior Loader for grasp optimization.

This module provides utilities for loading and applying grasp priors:
- Per-batch-item hand poses (translation, rotation, joints)
- Object state (rigid body or articulated)
- Contact link specifications
- Object-centric coordinate transformations

Coordinate System Convention:
- Hand poses are defined relative to object frame (Hand_T_Object)
- Object pose is defined in world frame (Object_T_World)
- During optimization, object stays at origin, hand moves relative to it

Future Extensions:
- Multiple hands (list of hand configs)
- Articulated objects (object with joint states)

Author: GraspQP Team
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import roma
import torch


@dataclass
class ObjectState:
    """
    Object state specification.

    For rigid bodies: pose in world frame (translation + rotation)
    For articulated objects (future): also includes joint states

    Attributes:
        translation: (3,) position in world frame
        rotation: (4,) quaternion (w,x,y,z) or (3,3) rotation matrix
        joints: Dict mapping joint_name -> angle (for articulated objects, future)
        is_articulated: Whether object has joints (future extension)
    """

    translation: Optional[List[float]] = None
    rotation: Optional[Union[List[float], List[List[float]]]] = None
    joints: Optional[Dict[str, float]] = None  # Future: for articulated objects
    is_articulated: bool = False


@dataclass
class HandPrior:
    """
    A single hand prior specification.

    Hand pose is defined relative to object frame (Hand_T_Object).

    Attributes:
        name: Optional identifier (e.g., "right_hand", "left_hand")
        hand_type: Hand model type (e.g., "allegro", "shadow_hand")
        translation: (3,) position relative to object frame
        rotation: (4,) quaternion (w,x,y,z) or (3,3) rotation matrix
        joints: Dict mapping joint_name -> angle, or None for default
        contact_links: List of link names for contact sampling
        contact_mode: "uniform", "guided", or "constrained"
        contact_preference_weight: For guided mode, fraction from preferred links
    """

    name: Optional[str] = None
    hand_type: Optional[str] = None
    translation: Optional[List[float]] = None
    rotation: Optional[Union[List[float], List[List[float]]]] = None
    joints: Optional[Dict[str, float]] = None
    contact_links: Optional[List[str]] = None
    contact_mode: str = "uniform"
    contact_preference_weight: float = 0.8


# Alias for backward compatibility
GraspPrior = HandPrior


@dataclass
class ContactConfig:
    """
    Global contact sampling configuration.

    Attributes:
        mode: "uniform", "guided", or "constrained"
        links: List of link names for guided/constrained sampling
        preference_weight: For guided mode, fraction from preferred links
        min_fingers: Minimum number of fingers in each sample
        max_contacts_per_link: Maximum contacts per link
    """

    mode: str = "uniform"
    links: Optional[List[str]] = None
    preference_weight: float = 0.8
    min_fingers: int = 2
    max_contacts_per_link: int = 4


@dataclass
class PriorConfig:
    """
    Configuration for prior-based initialization.

    Supports two YAML formats:

    New format (recommended):
    ```yaml
    object:
      translation: [0, 0, 0]
      rotation: [1, 0, 0, 0]  # w,x,y,z
    hands:
      - translation: [0.1, 0, 0]
        rotation: [1, 0, 0, 0]
        joints: {...}
    ```

    Legacy format (backward compatible):
    ```yaml
    priors:
      - translation: [0.1, 0, 0]
        rotation: [1, 0, 0, 0]
        joints: {...}
    ```

    Attributes:
        object: Object state in world frame (default: origin with identity rotation)
        hands: List of HandPrior objects (hand poses relative to object)
        contact: Global contact sampling configuration
        jitter_translation: Standard deviation for translation jitter
        jitter_rotation: Standard deviation for rotation jitter (radians)
        jitter_joints: Standard deviation for joint angle jitter
        prior_weight: Weight for prior deviation energy term
    """

    object: ObjectState = field(default_factory=ObjectState)
    hands: List[HandPrior] = field(default_factory=list)
    contact: ContactConfig = field(default_factory=ContactConfig)
    jitter_translation: float = 0.02
    jitter_rotation: float = 0.1
    jitter_joints: float = 0.1
    prior_weight: float = 10.0

    @property
    def priors(self) -> List[HandPrior]:
        """Alias for backward compatibility."""
        return self.hands

    @priors.setter
    def priors(self, value: List[HandPrior]):
        """Alias for backward compatibility."""
        self.hands = value


class GraspPriorLoader:
    """
    Load and manage grasp priors for optimization.

    Supports:
    - JSON/YAML prior files
    - Object state (rigid body, future: articulated)
    - Per-batch-item hand priors
    - Automatic expansion with jitter for batch diversity
    - Object-centric coordinate transformation

    Example YAML format (new):
    ```yaml
    object:
      translation: [0.0, 0.0, 0.0]
      rotation: [1, 0, 0, 0]  # w,x,y,z quaternion

    hands:
      - name: right_hand
        translation: [0.0, 0.0, 0.1]
        rotation: [1, 0, 0, 0]
        joints:
          index_joint_0: 0.3
        contact_links: [index_link_3, thumb_link_3]
        contact_mode: constrained

    jitter_translation: 0.02
    jitter_rotation: 0.1
    prior_weight: 10.0
    ```

    Legacy format (backward compatible):
    ```yaml
    priors:
      - translation: [0.0, 0.0, 0.1]
        rotation: [1, 0, 0, 0]
        joints: {...}
    ```
    """

    @staticmethod
    def load_from_file(path: str) -> PriorConfig:
        """
        Load prior configuration from JSON or YAML file.

        Args:
            path: Path to prior file (.json or .yaml/.yml)

        Returns:
            PriorConfig object
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prior file not found: {path}")

        ext = os.path.splitext(path)[1].lower()

        if ext == ".json":
            with open(path, "r") as f:
                data = json.load(f)
        elif ext in [".yaml", ".yml"]:
            try:
                import yaml

                with open(path, "r") as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML files: pip install pyyaml")
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        return GraspPriorLoader._parse_config(data)

    @staticmethod
    def _parse_config(data: dict) -> PriorConfig:
        """Parse raw dict into PriorConfig."""
        # Parse object state (new format)
        object_data = data.get("object", {})
        object_state = ObjectState(
            translation=object_data.get("translation"),
            rotation=object_data.get("rotation"),
            joints=object_data.get("joints"),
            is_articulated=object_data.get("is_articulated", False),
        )

        # Parse hand priors - support both "hands" (new) and "priors" (legacy)
        hands_data = data.get("hands", data.get("priors", []))
        hands = []
        for p in hands_data:
            hands.append(
                HandPrior(
                    name=p.get("name"),
                    hand_type=p.get("hand_type"),
                    translation=p.get("translation"),
                    rotation=p.get("rotation", p.get("quaternion")),  # Support both keys
                    joints=p.get("joints"),
                    contact_links=p.get("contact_links"),
                    contact_mode=p.get("contact_mode", "uniform"),
                    contact_preference_weight=p.get("contact_preference_weight", 0.8),
                )
            )

        # Parse contact config
        contact_data = data.get("contact", {})
        contact = ContactConfig(
            mode=contact_data.get("mode", "uniform"),
            links=contact_data.get("links"),
            preference_weight=contact_data.get("preference_weight", 0.8),
            min_fingers=contact_data.get("min_fingers", 2),
            max_contacts_per_link=contact_data.get("max_contacts_per_link", 4),
        )

        return PriorConfig(
            object=object_state,
            hands=hands,
            contact=contact,
            jitter_translation=data.get("jitter_translation", 0.02),
            jitter_rotation=data.get("jitter_rotation", 0.1),
            jitter_joints=data.get("jitter_joints", 0.1),
            prior_weight=data.get("prior_weight", 10.0),
        )

    @staticmethod
    def expand_priors(
        config: PriorConfig,
        batch_size: int,
        hand_model,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Expand priors to match batch size with jitter for diversity.

        Args:
            config: PriorConfig with base priors
            batch_size: Target batch size
            hand_model: HandModel for joint info
            device: Torch device

        Returns:
            Dict with:
                - "translation": (B, 3) tensor - hand translation relative to object
                - "rotation_6d": (B, 6) tensor (rot6d format) - hand rotation relative to object
                - "joints": (B, n_dofs) tensor - hand joint angles
                - "object_state": (B, 7) tensor - object pose [x,y,z, qx,qy,qz,qw]
                - "contact_configs": List of ContactSamplingConfig per batch item
                - "prior_weight": float
        """
        from graspqp.core.contact_sampler import ContactSamplingConfig

        n_hands = len(config.hands)
        n_dofs = hand_model.n_dofs

        # Initialize hand outputs
        translations = torch.zeros(batch_size, 3, device=device)
        rotations = torch.zeros(batch_size, 3, 3, device=device)
        joints = torch.zeros(batch_size, n_dofs, device=device)
        contact_configs = []

        # Initialize object state (shared across batch, or could be per-batch in future)
        # Format: [x, y, z, qx, qy, qz, qw]
        object_state = torch.zeros(batch_size, 7, device=device)
        object_state[:, 6] = 1.0  # Default: identity quaternion (qw=1)

        # Set object state from config
        if config.object.translation is not None:
            obj_t = torch.tensor(config.object.translation, device=device, dtype=torch.float)
            object_state[:, :3] = obj_t.unsqueeze(0).expand(batch_size, -1)

        if config.object.rotation is not None:
            obj_r = torch.tensor(config.object.rotation, device=device, dtype=torch.float)
            if obj_r.shape == (4,):
                # Quaternion (w,x,y,z) -> (x,y,z,w) for internal storage
                object_state[:, 3:7] = obj_r[[1, 2, 3, 0]].unsqueeze(0).expand(batch_size, -1)
            elif obj_r.shape == (3, 3):
                # Rotation matrix -> quaternion
                q_xyzw = roma.rotmat_to_unitquat(obj_r.unsqueeze(0))[0]
                object_state[:, 3:7] = q_xyzw.unsqueeze(0).expand(batch_size, -1)

        # Fill hand priors (cycling if needed)
        for i in range(batch_size):
            if n_hands > 0:
                prior = config.hands[i % n_hands]
                is_repeated = i >= n_hands
            else:
                prior = HandPrior()
                is_repeated = True

            # Translation (hand relative to object)
            if prior.translation is not None:
                t = torch.tensor(prior.translation, device=device, dtype=torch.float)
            else:
                t = torch.zeros(3, device=device)

            # Add jitter for repeated priors
            if is_repeated:
                t = t + torch.randn(3, device=device) * config.jitter_translation

            translations[i] = t

            # Rotation (hand relative to object)
            if prior.rotation is not None:
                r = torch.tensor(prior.rotation, device=device, dtype=torch.float)
                if r.shape == (4,):
                    # Quaternion (w,x,y,z) -> rotation matrix
                    r_xyzw = r[[1, 2, 3, 0]]  # Convert to (x,y,z,w)
                    R = roma.unitquat_to_rotmat(r_xyzw.unsqueeze(0))[0]
                elif r.shape == (3, 3):
                    R = r
                else:
                    raise ValueError(f"Invalid rotation shape: {r.shape}")
            else:
                R = torch.eye(3, device=device)

            # Add rotation jitter for repeated priors
            if is_repeated and config.jitter_rotation > 0:
                # Small random rotation
                axis = torch.randn(3, device=device)
                axis = axis / axis.norm()
                angle = torch.randn(1, device=device) * config.jitter_rotation
                R_jitter = roma.rotvec_to_rotmat(axis * angle)
                R = R_jitter @ R

            rotations[i] = R

            # Joints
            j = hand_model.default_state.clone().to(device)
            if prior.joints is not None:
                for joint_name, angle in prior.joints.items():
                    if joint_name in hand_model._actuated_joints_names:
                        idx = hand_model._actuated_joints_names.index(joint_name)
                        j[idx] = angle

            # Add joint jitter for repeated priors
            if is_repeated:
                j = j + torch.randn_like(j) * config.jitter_joints
                j = j.clamp(hand_model.joints_lower, hand_model.joints_upper)

            joints[i] = j

            # Contact config
            if prior.contact_links:
                contact_configs.append(
                    ContactSamplingConfig(
                        mode=prior.contact_mode,
                        preferred_links=prior.contact_links,
                        preference_weight=prior.contact_preference_weight,
                        min_fingers=2,
                    )
                )
            else:
                contact_configs.append(ContactSamplingConfig(mode="uniform"))

        # Convert rotation matrices to rot6d
        rotation_6d = rotations.transpose(1, 2)[:, :2].reshape(batch_size, 6)

        return {
            "translation": translations,
            "rotation_6d": rotation_6d,
            "joints": joints,
            "object_state": object_state,
            "contact_configs": contact_configs,
            "prior_weight": config.prior_weight,
        }

    @staticmethod
    def create_hand_pose_from_priors(
        prior_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Create hand_pose tensor from expanded priors.

        Args:
            prior_data: Output from expand_priors()

        Returns:
            (B, 3+6+n_dofs) hand_pose tensor
        """
        return torch.cat(
            [
                prior_data["translation"],
                prior_data["rotation_6d"],
                prior_data["joints"],
            ],
            dim=1,
        )

    @staticmethod
    def create_object_state_from_priors(
        prior_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Create object_state tensor from expanded priors.

        Args:
            prior_data: Output from expand_priors()

        Returns:
            (B, 7) object_state tensor [x, y, z, qx, qy, qz, qw]
        """
        return prior_data["object_state"]

    @staticmethod
    def get_default_object_state(batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Create default object state at origin with identity rotation.

        Args:
            batch_size: Number of batch items
            device: Torch device

        Returns:
            (B, 7) object_state tensor [x, y, z, qx, qy, qz, qw]
        """
        object_state = torch.zeros(batch_size, 7, device=device)
        object_state[:, 6] = 1.0  # qw = 1 (identity quaternion)
        return object_state


def compute_prior_energy(
    hand_pose: torch.Tensor,
    prior_pose: torch.Tensor,
    prior_weight: float = 10.0,
) -> torch.Tensor:
    """
    Compute energy term penalizing deviation from prior pose.

    Args:
        hand_pose: (B, 3+6+n_dofs) current hand pose
        prior_pose: (B, 3+6+n_dofs) prior hand pose
        prior_weight: Weight for prior deviation

    Returns:
        (B,) energy values
    """
    from graspqp.utils.transforms import robust_compute_rotation_matrix_from_ortho6d

    # Translation deviation
    E_trans = ((hand_pose[:, :3] - prior_pose[:, :3]) ** 2).sum(-1)

    # Rotation deviation (geodesic distance)
    R_current = robust_compute_rotation_matrix_from_ortho6d(hand_pose[:, 3:9])
    R_prior = robust_compute_rotation_matrix_from_ortho6d(prior_pose[:, 3:9])

    # Geodesic distance: arccos((trace(R1^T R2) - 1) / 2)
    R_diff = R_current.transpose(1, 2) @ R_prior
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    E_rot = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))

    # Joint deviation (smaller weight)
    E_joints = ((hand_pose[:, 9:] - prior_pose[:, 9:]) ** 2).sum(-1) * 0.1

    return prior_weight * (E_trans + E_rot + E_joints)


class ObjectCentricTransform:
    """
    Utilities for object-centric coordinate transformations.

    Transforms hand poses between world frame and object frame.
    Object frame is centered at CoG with identity rotation.
    """

    @staticmethod
    def world_to_object(
        hand_pose: torch.Tensor,
        object_cog: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transform hand pose from world frame to object frame.

        Args:
            hand_pose: (B, 3+6+n_dofs) in world frame
            object_cog: (B, 3) object center of gravity

        Returns:
            (B, 3+6+n_dofs) in object frame
        """
        result = hand_pose.clone()
        # Translate relative to object CoG
        result[:, :3] = hand_pose[:, :3] - object_cog
        # Rotation and joints unchanged (object has identity rotation)
        return result

    @staticmethod
    def object_to_world(
        hand_pose: torch.Tensor,
        object_cog: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transform hand pose from object frame to world frame.

        Args:
            hand_pose: (B, 3+6+n_dofs) in object frame
            object_cog: (B, 3) object center of gravity

        Returns:
            (B, 3+6+n_dofs) in world frame
        """
        result = hand_pose.clone()
        # Translate back to world
        result[:, :3] = hand_pose[:, :3] + object_cog
        return result
