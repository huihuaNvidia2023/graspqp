"""
Optimization context providing shared resources for all costs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

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
    - Fixed contact indices
    - Lazy cache for intermediate results

    Attributes:
        hand_model: The hand model (Allegro, MANO, etc.)
        object_model: The object model with SDF
        reference: The reference trajectory from video
        contact_indices: Fixed contact indices for the trajectory
        device: Torch device
    """

    def __init__(
        self,
        hand_model: Any,
        object_model: Any,
        reference: Optional["ReferenceTrajectory"] = None,
        device: Optional[torch.device] = None,
    ):
        self.hand_model = hand_model
        self.object_model = object_model
        self.reference = reference
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cache for intermediate results
        self._cache: Dict[str, Any] = {}
        self._cache_scopes: Dict[str, str] = {}  # key -> scope

        # Fixed contact indices (from reference)
        self._contact_indices: Optional[Tensor] = None
        if reference is not None:
            self._contact_indices = reference.contact_indices

    @property
    def contact_indices(self) -> Optional[Tensor]:
        """Get fixed contact indices."""
        return self._contact_indices

    def set_contact_indices(self, indices: Tensor):
        """Set fixed contact indices (called once at initialization)."""
        self._contact_indices = indices

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
        # Set hand parameters with fixed contact indices
        self.hand_model.set_parameters(hand_states, contact_point_indices=self._contact_indices)
        return self.hand_model.contact_points

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
