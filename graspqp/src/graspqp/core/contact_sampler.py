"""
Hierarchical Contact Sampler for efficient grasp optimization.

This module provides intelligent contact point sampling that:
1. Ensures finger diversity (contacts from multiple fingers)
2. Supports guided and constrained sampling modes
3. Uses hierarchical sampling: first select links, then sample points

Author: GraspQP Team
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class ContactSamplingConfig:
    """
    Configuration for contact sampling behavior.

    Attributes:
        mode: Sampling mode - "uniform", "guided", or "constrained"
            - uniform: Sample uniformly from all contact candidates
            - guided: Prefer specified links but allow exploration
            - constrained: Only sample from specified links
        preferred_links: List of link names to prefer/constrain to
        preference_weight: For guided mode, fraction of samples from preferred links
        min_fingers: Minimum number of different fingers to use
        max_contacts_per_link: Maximum contact points from a single link
        link_weights: Optional per-link sampling weights
    """

    mode: str = "uniform"
    preferred_links: Optional[List[str]] = None
    preference_weight: float = 0.8
    min_fingers: int = 2
    max_contacts_per_link: int = 2
    link_weights: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.mode not in ["uniform", "guided", "constrained"]:
            raise ValueError(f"Invalid mode: {self.mode}. " f"Must be 'uniform', 'guided', or 'constrained'")

        if self.mode in ["guided", "constrained"] and not self.preferred_links:
            raise ValueError(f"Mode '{self.mode}' requires preferred_links to be specified")


class HierarchicalContactSampler:
    """
    Efficient contact sampling with finger diversity guarantees.

    Instead of sampling uniformly from a flat pool of contact candidates,
    this sampler uses a hierarchical approach:

    1. Group contact points by link (finger segment)
    2. Sample which links to use (ensuring finger diversity)
    3. Sample contact points within selected links

    This approach:
    - Guarantees contacts from multiple fingers (configurable)
    - Reduces wasted samples on correlated configurations
    - Aligns with how humans think about grasping

    Example:
        >>> config = ContactSamplingConfig(
        ...     mode="constrained",
        ...     preferred_links=["index_link_3", "thumb_link_3"],
        ...     min_fingers=2
        ... )
        >>> sampler = HierarchicalContactSampler(hand_model, config)
        >>> contacts = sampler.sample(batch_size=32, n_contacts=12)
    """

    # Known finger prefixes for different hand conventions
    FINGER_PREFIXES = [
        "index",
        "middle",
        "ring",
        "thumb",
        "pinky",
        "little",
        "ff",
        "mf",
        "rf",
        "th",
        "lf",  # Shadow hand style
        "finger_0",
        "finger_1",
        "finger_2",
        "finger_3",  # Generic
    ]

    # Links that should NOT be counted as fingers for diversity
    NON_FINGER_LINKS = ["palm", "base", "wrist", "hand"]

    def __init__(self, hand_model, config: ContactSamplingConfig):
        """
        Initialize the hierarchical contact sampler.

        Args:
            hand_model: HandModel instance with mesh and contact info
            config: ContactSamplingConfig specifying sampling behavior
        """
        self.hand_model = hand_model
        self.config = config
        self.device = hand_model.device

        # Build index structures
        self._build_link_index()

        # Validate configuration
        self._validate_config()

    def _build_link_index(self):
        """Build mappings between links, fingers, and contact indices."""
        self.link_to_indices: Dict[str, torch.Tensor] = {}
        self.link_to_finger: Dict[str, str] = {}
        self.finger_to_links: Dict[str, List[str]] = {}

        current_idx = 0
        for link_name in self.hand_model.mesh:
            n_contacts = len(self.hand_model.mesh[link_name]["contact_candidates"])
            if n_contacts == 0:
                continue

            # Store indices for this link
            indices = list(range(current_idx, current_idx + n_contacts))
            self.link_to_indices[link_name] = torch.tensor(indices, dtype=torch.long, device=self.device)

            # Extract finger name
            finger = self._extract_finger_name(link_name)
            self.link_to_finger[link_name] = finger

            # Group links by finger
            if finger not in self.finger_to_links:
                self.finger_to_links[finger] = []
            self.finger_to_links[finger].append(link_name)

            current_idx += n_contacts

        # Precompute allowed links for guided/constrained modes
        if self.config.preferred_links:
            self.allowed_links = [link for link in self.config.preferred_links if link in self.link_to_indices]
            if len(self.allowed_links) == 0:
                raise ValueError(
                    f"None of the preferred_links exist in hand model. " f"Available links: {list(self.link_to_indices.keys())}"
                )
        else:
            self.allowed_links = list(self.link_to_indices.keys())

        # Compute total available contacts for allowed links
        self.n_allowed_contacts = sum(len(self.link_to_indices[link]) for link in self.allowed_links)

    def _validate_config(self):
        """Validate configuration against hand model."""
        if self.config.mode == "constrained":
            # Check we have enough fingers
            allowed_fingers = set(self.link_to_finger[link] for link in self.allowed_links)
            if len(allowed_fingers) < self.config.min_fingers:
                print(
                    f"Warning: Only {len(allowed_fingers)} fingers available in "
                    f"preferred_links, but min_fingers={self.config.min_fingers}. "
                    f"Reducing min_fingers to {len(allowed_fingers)}."
                )
                self.config.min_fingers = len(allowed_fingers)

    def _extract_finger_name(self, link_name: str) -> str:
        """
        Extract finger identifier from link name.

        Handles various naming conventions:
        - "index_link_3" → "index"
        - "thumb_joint_2" → "thumb"
        - "FFtip" → "ff"
        - "finger_0_link_2" → "finger_0"
        - "palm_link" → "palm" (non-finger)
        """
        link_lower = link_name.lower()

        # Check for non-finger links first
        for non_finger in self.NON_FINGER_LINKS:
            if non_finger in link_lower:
                return non_finger  # Return as-is (not counted for diversity)

        for prefix in self.FINGER_PREFIXES:
            if prefix in link_lower:
                # Handle numbered fingers (finger_0, finger_1, etc.)
                if prefix.startswith("finger_"):
                    # Find the full finger_N pattern
                    import re

                    match = re.search(r"finger_\d+", link_lower)
                    if match:
                        return match.group()
                return prefix

        # Fallback: use first part before underscore
        return link_name.split("_")[0].lower()

    def _index_to_link_name(self, idx: int) -> str:
        """Map global contact index back to link name."""
        for link_name, indices in self.link_to_indices.items():
            if idx in indices.tolist():
                return link_name
        return "unknown"

    def _link_priority(self, link_name: str) -> int:
        """
        Compute priority for a link. Higher = more likely to sample.

        Prefers distal links (fingertips) over proximal links.
        """
        link_lower = link_name.lower()

        # Explicit tip naming
        if "tip" in link_lower:
            return 10

        # Numbered links (link_3 > link_2 > link_1)
        # This line loops from 5 down to 0 (inclusive), i.e., i=5,4,3,2,1,0 (most distal to most proximal)
        for i in range(5, -1, -1):
            if f"link_{i}" in link_lower or f"_{i}" in link_lower:
                return i
            if f"l{i}" in link_lower:  # Abbreviated form
                return i

        return 0

    def sample(self, batch_size: int, n_contacts: int) -> torch.Tensor:
        """
        Sample contact point indices.

        Args:
            batch_size: Number of grasp configurations to sample
            n_contacts: Number of contact points per configuration

        Returns:
            Tensor of shape (batch_size, n_contacts) with contact indices
        """
        if self.config.mode == "uniform":
            return self._sample_uniform(batch_size, n_contacts)
        elif self.config.mode == "guided":
            return self._sample_guided(batch_size, n_contacts)
        elif self.config.mode == "constrained":
            return self._sample_constrained(batch_size, n_contacts)
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

    def _sample_uniform(self, batch_size: int, n_contacts: int) -> torch.Tensor:
        """Original uniform sampling (baseline)."""
        return torch.randint(self.hand_model.n_contact_candidates, (batch_size, n_contacts), device=self.device)

    def _sample_hierarchical(self, batch_size: int, n_contacts: int, link_pool: List[str]) -> torch.Tensor:
        """
        Core hierarchical sampling logic.

        1. Select links with finger diversity guarantee
        2. Sample contact points within selected links
        """
        result = torch.zeros(batch_size, n_contacts, dtype=torch.long, device=self.device)

        for b in range(batch_size):
            # Step 1: Select links ensuring finger diversity
            selected = self._select_links_with_diversity(link_pool, n_contacts)

            # Step 2: Sample from each selected link
            contact_idx = 0
            for link_name, n_pts in selected.items():
                link_indices = self.link_to_indices[link_name]
                n_available = len(link_indices)

                # Sample without replacement within link
                if n_pts <= n_available:
                    perm = torch.randperm(n_available, device=self.device)[:n_pts]
                    sampled = link_indices[perm]
                else:
                    # Need to sample with replacement (shouldn't happen with proper config)
                    sampled = link_indices[torch.randint(n_available, (n_pts,), device=self.device)]

                result[b, contact_idx : contact_idx + n_pts] = sampled
                contact_idx += n_pts

        return result

    def _select_links_with_diversity(self, link_pool: List[str], n_contacts: int) -> Dict[str, int]:
        """
        Select links ensuring finger diversity.

        Returns dict mapping link_name → number of contacts to sample.

        Non-finger links (palm, base, wrist) are not counted toward
        finger diversity but can still provide contacts.
        """
        # Group available links by finger (excluding non-finger links)
        available_by_finger: Dict[str, List[str]] = {}
        non_finger_links: List[str] = []

        for link in link_pool:
            finger = self.link_to_finger.get(link, "other")

            # Check if this is a non-finger link
            if finger in self.NON_FINGER_LINKS:
                non_finger_links.append(link)
            else:
                if finger not in available_by_finger:
                    available_by_finger[finger] = []
                available_by_finger[finger].append(link)

        fingers = list(available_by_finger.keys())
        n_fingers = len(fingers)
        min_fingers = min(self.config.min_fingers, n_fingers)

        # Select which fingers to use
        if min_fingers > 0 and n_fingers > 0:
            selected_fingers = random.sample(fingers, min_fingers)
        else:
            selected_fingers = []

        result: Dict[str, int] = {}
        remaining_contacts = n_contacts

        # Calculate max_contacts_per_link dynamically if needed
        # to ensure we can sample all requested contacts
        total_available_in_pool = sum(len(self.link_to_indices[link]) for link in link_pool)
        n_links_in_pool = len(link_pool)

        # If strict max_contacts_per_link would prevent sampling enough,
        # increase it dynamically
        effective_max_per_link = self.config.max_contacts_per_link
        if n_links_in_pool > 0:
            min_needed_per_link = (n_contacts + n_links_in_pool - 1) // n_links_in_pool
            effective_max_per_link = max(effective_max_per_link, min_needed_per_link)

        # First pass: distribute across selected fingers (ensuring diversity)
        for finger in selected_fingers:
            links = available_by_finger[finger]
            # Sort by priority (prefer distal/tip links)
            links_sorted = sorted(links, key=self._link_priority, reverse=True)

            for link in links_sorted:
                if remaining_contacts <= 0:
                    break

                n_available = len(self.link_to_indices[link])
                n_from_link = min(effective_max_per_link, n_available, remaining_contacts)

                if n_from_link > 0:
                    result[link] = n_from_link
                    remaining_contacts -= n_from_link

        # Second pass: if still need more, sample from remaining finger links
        if remaining_contacts > 0:
            remaining_finger_links = [
                link for finger_links in available_by_finger.values() for link in finger_links if link not in result
            ]
            random.shuffle(remaining_finger_links)

            for link in remaining_finger_links:
                if remaining_contacts <= 0:
                    break

                n_available = len(self.link_to_indices[link])
                current = result.get(link, 0)
                additional = min(effective_max_per_link - current, n_available - current, remaining_contacts)

                if additional > 0:
                    result[link] = current + additional
                    remaining_contacts -= additional

        # Third pass: use non-finger links if still need more (and they're in pool)
        if remaining_contacts > 0:
            random.shuffle(non_finger_links)
            for link in non_finger_links:
                if remaining_contacts <= 0:
                    break

                n_available = len(self.link_to_indices[link])
                current = result.get(link, 0)
                additional = min(effective_max_per_link - current, n_available - current, remaining_contacts)

                if additional > 0:
                    result[link] = current + additional
                    remaining_contacts -= additional

        return result

    def _sample_guided(self, batch_size: int, n_contacts: int) -> torch.Tensor:
        """
        Guided sampling: prefer specified links but allow exploration.

        Samples preference_weight fraction from preferred links,
        (1 - preference_weight) fraction from all links for exploration.
        """
        n_preferred = int(n_contacts * self.config.preference_weight)
        n_explore = n_contacts - n_preferred

        # Ensure at least 1 exploration point if n_contacts > 1
        if n_contacts > 1 and n_explore == 0:
            n_explore = 1
            n_preferred = n_contacts - 1

        # Sample from preferred links
        preferred = self._sample_hierarchical(batch_size, n_preferred, self.allowed_links)

        # Sample from all links for exploration
        all_links = list(self.link_to_indices.keys())
        explore = self._sample_hierarchical(batch_size, n_explore, all_links)

        # Combine
        result = torch.cat([preferred, explore], dim=1)

        # Shuffle within each batch to mix preferred and exploration
        for b in range(batch_size):
            perm = torch.randperm(n_contacts, device=self.device)
            result[b] = result[b, perm]

        return result

    def _sample_constrained(self, batch_size: int, n_contacts: int) -> torch.Tensor:
        """
        Constrained sampling: only sample from preferred links.

        Raises ValueError if more contacts requested than available.
        """
        if n_contacts > self.n_allowed_contacts:
            raise ValueError(
                f"Requested {n_contacts} contacts but only "
                f"{self.n_allowed_contacts} available in allowed links: "
                f"{self.allowed_links}"
            )

        return self._sample_hierarchical(batch_size, n_contacts, self.allowed_links)

    def get_link_info(self) -> Dict[str, Dict]:
        """
        Get information about links for debugging/visualization.

        Returns dict with link statistics.
        """
        info = {}
        for link_name in self.link_to_indices:
            info[link_name] = {
                "n_contacts": len(self.link_to_indices[link_name]),
                "finger": self.link_to_finger[link_name],
                "priority": self._link_priority(link_name),
                "is_allowed": link_name in self.allowed_links,
            }
        return info


def create_sampler_from_args(hand_model, args) -> HierarchicalContactSampler:
    """
    Factory function to create sampler from command line arguments.

    Args:
        hand_model: HandModel instance
        args: Namespace with contact_mode, contact_links, etc.

    Returns:
        Configured HierarchicalContactSampler
    """
    preferred_links = None
    if hasattr(args, "contact_links") and args.contact_links:
        preferred_links = [l.strip() for l in args.contact_links.split(",")]

    config = ContactSamplingConfig(
        mode=getattr(args, "contact_mode", "uniform"),
        preferred_links=preferred_links,
        preference_weight=getattr(args, "contact_preference_weight", 0.8),
        min_fingers=getattr(args, "min_fingers", 2),
        max_contacts_per_link=getattr(args, "max_contacts_per_link", 2),
    )

    return HierarchicalContactSampler(hand_model, config)
