"""
Optimization context providing shared resources for all costs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .state import ReferenceTrajectory


class OptimizationContext:
    """
    Shared context for all cost functions during optimization.

    Provides:
    - Hand and object models
    - Reference trajectory (the video data)
    - Contact configuration (which fingers, how many points)
    - Contact sampler for sampling contact points within allowed fingers
    - Lazy cache for intermediate results

    Contact Model:
        - Reference specifies which FINGERS are in contact (high-level)
        - Optimizer determines which specific CONTACT POINTS on those fingers
        - Contact sampler samples points respecting finger constraints
        - n_contacts specifies minimum contact points required

    Attributes:
        hand_model: The hand model (Allegro, MANO, etc.)
        object_model: The object model with SDF
        reference: The reference trajectory from video
        contact_sampler: Sampler for contact points (respects finger constraints)
        device: Torch device
    """

    def __init__(
        self,
        hand_model: Any,
        object_model: Any,
        reference: Optional["ReferenceTrajectory"] = None,
        contact_sampler: Any = None,  # HierarchicalContactSampler
        device: Optional[torch.device] = None,
    ):
        self.hand_model = hand_model
        self.object_model = object_model
        self.reference = reference
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cache for intermediate results
        self._cache: Dict[str, Any] = {}
        self._cache_scopes: Dict[str, str] = {}  # key -> scope

        # Contact sampling
        self._contact_sampler = contact_sampler
        self._contact_fingers: Optional[List[str]] = None
        self._n_contacts: int = 8

        if reference is not None:
            self._contact_fingers = reference.contact_fingers
            self._n_contacts = reference.n_contacts

        # Current contact indices (updated by optimizer during sampling)
        self._current_contact_indices: Optional[Tensor] = None

        # Flag to skip set_parameters (for gradient computation mode)
        # When True, costs should assume hand_model is already configured
        self._skip_set_parameters: bool = False

    @property
    def contact_fingers(self) -> Optional[List[str]]:
        """Get list of fingers that should be in contact."""
        return self._contact_fingers

    @property
    def n_contacts(self) -> int:
        """Get minimum number of contact points required."""
        return self._n_contacts

    @property
    def contact_sampler(self):
        """Get contact point sampler (respects finger constraints)."""
        return self._contact_sampler

    def set_contact_sampler(self, sampler: Any):
        """Set contact point sampler."""
        self._contact_sampler = sampler

    @property
    def contact_indices(self) -> Optional[Tensor]:
        """Get current contact point indices (set by optimizer)."""
        return self._current_contact_indices

    def set_contact_indices(self, indices: Tensor):
        """Set current contact indices (called by optimizer during step)."""
        self._current_contact_indices = indices

    def create_contact_sampler_from_reference(self):
        """
        Create a contact sampler based on reference finger constraints.

        This creates a HierarchicalContactSampler that only samples
        contact points from the fingers specified in the reference.
        """
        if self._contact_fingers is None:
            # No finger constraints - use uniform sampling
            return None

        try:
            from graspqp.core import ContactSamplingConfig, HierarchicalContactSampler

            config = ContactSamplingConfig(
                mode="guided",
                preferred_links=self._contact_fingers,
                preference_weight=1.0,  # Only sample from these fingers
                min_fingers=len(self._contact_fingers),
            )
            self._contact_sampler = HierarchicalContactSampler(self.hand_model, config)
            return self._contact_sampler
        except ImportError:
            # Fallback if graspqp.core not available
            return None

    def get_cached(self, key: str) -> Optional[Any]:
        """Get a cached value if it exists."""
        return self._cache.get(key)

    def set_cached(self, key: str, value: Any, scope: str = "step"):
        """
        Set a cached value.

        Args:
            key: Cache key
            value: Value to cache
            scope: Cache scope - "step" (cleared each step),
                   "trajectory" (valid for entire optimization),
                   "persistent" (never cleared)
        """
        self._cache[key] = value
        self._cache_scopes[key] = scope

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        scope: str = "step",
    ) -> Any:
        """
        Get cached value or compute and cache it.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            scope: Cache scope

        Returns:
            Cached or computed value
        """
        if key in self._cache:
            return self._cache[key]

        value = compute_fn()
        self._cache[key] = value
        self._cache_scopes[key] = scope
        return value

    def clear_step_cache(self):
        """Clear cache entries with scope='step'. Called at start of each optimization step."""
        keys_to_remove = [key for key, scope in self._cache_scopes.items() if scope == "step"]
        for key in keys_to_remove:
            del self._cache[key]
            del self._cache_scopes[key]

    def clear_all_cache(self):
        """Clear entire cache."""
        self._cache.clear()
        self._cache_scopes.clear()

    def get_contact_points(self, hand_states: Tensor) -> Tensor:
        """
        Compute contact points for given hand states.

        Args:
            hand_states: Hand states, shape (N, D_hand) where N = B*T or B*K*T

        Returns:
            Contact points, shape (N, n_contacts, 3)
        """
        # Set hand parameters with current contact indices (if set by optimizer)
        if self._current_contact_indices is not None:
            self.hand_model.set_parameters(hand_states, contact_point_indices=self._current_contact_indices)
        else:
            self.hand_model.set_parameters(hand_states)
        return self.hand_model.contact_points

    def sample_contact_indices(self, batch_size: int) -> Tensor:
        """
        Sample contact point indices respecting finger constraints.

        Args:
            batch_size: Number of samples to generate

        Returns:
            Contact indices, shape (batch_size, n_contacts)
        """
        if self._contact_sampler is not None:
            return self._contact_sampler.sample(batch_size, self._n_contacts)
        else:
            # Fallback: uniform sampling from all contact candidates
            n_candidates = self.hand_model.n_contact_candidates
            return torch.randint(
                n_candidates,
                size=(batch_size, self._n_contacts),
                device=self.device,
            )

    def compute_sdf(self, points: Tensor) -> Tensor:
        """
        Compute signed distance field values for points.

        Args:
            points: Points to query, shape (N, M, 3)

        Returns:
            SDF values, shape (N, M)
        """
        distance, _ = self.object_model.cal_distance(points)
        return distance

    # ========================================================================
    # Hand Model Caching - avoid redundant FK computation
    # ========================================================================

    def ensure_hand_configured(self, flat_hand: Tensor) -> None:
        """
        Ensure hand model is configured with given pose, skipping if already set.

        This avoids redundant FK computation when multiple costs call this
        with the same hand state in the same optimization step.

        When _skip_set_parameters is True (gradient mode), we skip calling
        set_parameters but still recompute FK to ensure gradients flow through.

        Args:
            flat_hand: Flattened hand states, shape (N, D_hand)
        """
        cache_key = "_hand_configured_ptr"
        cached_ptr = self.get_cached(cache_key)

        # Fast identity check using data pointer
        current_ptr = flat_hand.data_ptr()

        if cached_ptr != current_ptr:
            if self._skip_set_parameters:
                # In gradient mode: update translation, rotation, and FK
                # without calling set_parameters (which clones the tensor).
                # This preserves gradient flow through hand_pose.
                from graspqp.utils.transforms import robust_compute_rotation_matrix_from_ortho6d

                self.hand_model.global_translation = flat_hand[:, :3]
                self.hand_model.global_rotation = robust_compute_rotation_matrix_from_ortho6d(flat_hand[:, 3:9])
                self.hand_model.current_status = self.hand_model.fk(flat_hand[:, 9:])
            else:
                # Normal mode: full set_parameters
                if self._current_contact_indices is not None:
                    self.hand_model.set_parameters(flat_hand, contact_point_indices=self._current_contact_indices)
                else:
                    self.hand_model.set_parameters(flat_hand)
            self.set_cached(cache_key, current_ptr, scope="step")

    def get_contact_points_cached(self, flat_hand: Tensor) -> Tensor:
        """
        Get contact points, using cache if hand state unchanged.

        Args:
            flat_hand: Flattened hand states, shape (N, D_hand)

        Returns:
            Contact points, shape (N, n_contacts, 3)
        """
        self.ensure_hand_configured(flat_hand)
        return self.hand_model.contact_points

    def get_surface_points_cached(self, flat_hand: Tensor, n_subsample: Optional[int] = None) -> Tensor:
        """
        Get surface points, using cache if hand state unchanged.

        Args:
            flat_hand: Flattened hand states, shape (N, D_hand)
            n_subsample: If set, subsample to this many points (cached per step)

        Returns:
            Surface points, shape (N, n_pts, 3)
        """
        self.ensure_hand_configured(flat_hand)

        cache_key = "_surface_points"
        if n_subsample is not None:
            cache_key = f"_surface_points_{n_subsample}"

        cached = self.get_cached(cache_key)
        if cached is not None:
            return cached

        # Compute surface points
        surface_points = self.hand_model.get_surface_points()

        # Subsample if needed
        if n_subsample is not None and surface_points.shape[1] > n_subsample:
            idx = torch.randperm(surface_points.shape[1], device=surface_points.device)[:n_subsample]
            surface_points = surface_points[:, idx]

        self.set_cached(cache_key, surface_points, scope="step")
        return surface_points
