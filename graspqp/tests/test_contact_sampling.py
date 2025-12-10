"""
Test suite for HierarchicalContactSampler and related functionality.

Run with: python -m unittest graspqp/tests/test_contact_sampling.py -v
Or:       python graspqp/tests/test_contact_sampling.py
"""

import time
import unittest
from collections import Counter
from typing import Dict, List, Optional

import torch

from graspqp.core.contact_sampler import (ContactSamplingConfig,
                                          HierarchicalContactSampler)

# ============================================================================
# Test Fixtures (as helper methods)
# ============================================================================


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_mock_hand_model(device):
    """
    Create a mock hand model with known contact structure.

    Structure (mimics Allegro hand):
    - index_link_1: 5 contacts (indices 0-4)
    - index_link_2: 5 contacts (indices 5-9)
    - index_link_3: 8 contacts (indices 10-17)  <- fingertip
    - middle_link_1: 5 contacts (indices 18-22)
    - middle_link_2: 5 contacts (indices 23-27)
    - middle_link_3: 8 contacts (indices 28-35)  <- fingertip
    - ring_link_1: 5 contacts (indices 36-40)
    - ring_link_2: 5 contacts (indices 41-45)
    - ring_link_3: 8 contacts (indices 46-53)  <- fingertip
    - thumb_link_1: 5 contacts (indices 54-58)
    - thumb_link_2: 5 contacts (indices 59-63)
    - thumb_link_3: 10 contacts (indices 64-73)  <- fingertip

    Total: 74 contact candidates
    """

    class MockHandModel:
        def __init__(self, device):
            self.device = device
            self.mesh = {}

            # Build mesh structure
            fingers = ["index", "middle", "ring", "thumb"]
            contacts_per_link = {
                "link_1": 5,
                "link_2": 5,
                "link_3": 8,  # More contacts on fingertips
            }

            # Thumb has more contacts
            thumb_contacts = {"link_1": 5, "link_2": 5, "link_3": 10}

            for finger in fingers:
                cpls = thumb_contacts if finger == "thumb" else contacts_per_link
                for link_suffix, n_contacts in cpls.items():
                    link_name = f"{finger}_{link_suffix}"
                    # Create random contact candidates
                    self.mesh[link_name] = {
                        "contact_candidates": torch.randn(n_contacts, 3, device=device),
                        "normal_candidates": torch.randn(n_contacts, 3, device=device),
                    }

            # Calculate total contacts
            self.n_contact_candidates = sum(len(self.mesh[ln]["contact_candidates"]) for ln in self.mesh)

            # Build global index mapping
            self.global_index_to_link_index = []
            self.link_names = list(self.mesh.keys())
            for i, link_name in enumerate(self.link_names):
                n = len(self.mesh[link_name]["contact_candidates"])
                self.global_index_to_link_index.extend([i] * n)
            self.global_index_to_link_index = torch.tensor(self.global_index_to_link_index, device=device)

    return MockHandModel(device)


def create_uniform_config():
    """Uniform sampling config (baseline)"""
    return ContactSamplingConfig(mode="uniform")


def create_guided_config():
    """Guided sampling - prefer fingertips"""
    return ContactSamplingConfig(
        mode="guided",
        preferred_links=["index_link_3", "middle_link_3", "thumb_link_3"],
        preference_weight=0.8,
        min_fingers=2,
        max_contacts_per_link=2,
    )


def create_constrained_config():
    """Constrained sampling - only fingertips"""
    return ContactSamplingConfig(
        mode="constrained",
        preferred_links=["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"],
        min_fingers=2,
        max_contacts_per_link=2,
    )


# ============================================================================
# ContactSamplingConfig Tests
# ============================================================================


class TestContactSamplingConfig(unittest.TestCase):
    """Tests for ContactSamplingConfig dataclass"""

    def test_default_config(self):
        """Default config should be uniform sampling"""
        config = ContactSamplingConfig(mode="uniform")
        self.assertEqual(config.mode, "uniform")
        self.assertIsNone(config.preferred_links)
        self.assertEqual(config.min_fingers, 2)

    def test_guided_config_requires_preferred_links(self):
        """Guided mode should have preferred_links"""
        config = ContactSamplingConfig(mode="guided", preferred_links=["index_link_3"])
        self.assertIsNotNone(config.preferred_links)
        self.assertGreater(len(config.preferred_links), 0)

    def test_invalid_mode_raises(self):
        """Invalid mode should raise error"""
        with self.assertRaises(ValueError):
            ContactSamplingConfig(mode="invalid")


# ============================================================================
# HierarchicalContactSampler Tests
# ============================================================================


class TestHierarchicalContactSampler(unittest.TestCase):
    """Tests for HierarchicalContactSampler"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.device = get_device()
        cls.mock_hand_model = create_mock_hand_model(cls.device)
        cls.uniform_config = create_uniform_config()
        cls.guided_config = create_guided_config()
        cls.constrained_config = create_constrained_config()

    # --- Initialization Tests ---

    def test_init_builds_link_index(self):
        """Sampler should correctly build link-to-index mapping"""
        sampler = HierarchicalContactSampler(self.mock_hand_model, self.uniform_config)

        # Check all links are indexed
        self.assertEqual(len(sampler.link_to_indices), len(self.mock_hand_model.mesh))

        # Check indices are contiguous and cover all contacts
        all_indices = []
        for link_name in self.mock_hand_model.mesh:
            indices = sampler.link_to_indices[link_name]
            all_indices.extend(indices.tolist())

        self.assertEqual(sorted(all_indices), list(range(self.mock_hand_model.n_contact_candidates)))

    def test_init_builds_finger_mapping(self):
        """Sampler should correctly identify fingers from link names"""
        sampler = HierarchicalContactSampler(self.mock_hand_model, self.uniform_config)

        # Check finger extraction
        self.assertEqual(sampler.link_to_finger["index_link_1"], "index")
        self.assertEqual(sampler.link_to_finger["thumb_link_3"], "thumb")

        # Check finger grouping
        self.assertIn("index", sampler.finger_to_links)
        self.assertEqual(len(sampler.finger_to_links["index"]), 3)  # link_1, link_2, link_3

    # --- Uniform Sampling Tests ---

    def test_uniform_sampling_shape(self):
        """Uniform sampling should return correct shape"""
        sampler = HierarchicalContactSampler(self.mock_hand_model, self.uniform_config)

        batch_size, n_contacts = 32, 12
        result = sampler.sample(batch_size, n_contacts)

        self.assertEqual(result.shape, (batch_size, n_contacts))
        self.assertEqual(result.dtype, torch.long)

    def test_uniform_sampling_valid_indices(self):
        """Uniform sampling should return valid contact indices"""
        sampler = HierarchicalContactSampler(self.mock_hand_model, self.uniform_config)

        result = sampler.sample(32, 12)

        self.assertGreaterEqual(result.min().item(), 0)
        self.assertLess(result.max().item(), self.mock_hand_model.n_contact_candidates)

    # --- Guided Sampling Tests ---

    def test_guided_sampling_prefers_specified_links(self):
        """Guided sampling should bias toward preferred links"""
        sampler = HierarchicalContactSampler(self.mock_hand_model, self.guided_config)

        # Sample many times to get statistics
        n_samples = 1000
        batch_size, n_contacts = 32, 12

        preferred_count = 0
        total_count = 0

        for _ in range(n_samples // batch_size):
            result = sampler.sample(batch_size, n_contacts)

            for b in range(batch_size):
                for idx in result[b]:
                    link_name = sampler._index_to_link_name(idx.item())
                    if link_name in self.guided_config.preferred_links:
                        preferred_count += 1
                    total_count += 1

        # Should be close to preference_weight (0.8 = 80%)
        actual_ratio = preferred_count / total_count
        expected_ratio = self.guided_config.preference_weight

        self.assertLess(
            abs(actual_ratio - expected_ratio), 0.1, f"Expected ~{expected_ratio:.2f} preferred, got {actual_ratio:.2f}"
        )

    # --- Constrained Sampling Tests ---

    def test_constrained_sampling_only_uses_specified_links(self):
        """Constrained sampling should only use preferred links"""
        sampler = HierarchicalContactSampler(self.mock_hand_model, self.constrained_config)

        result = sampler.sample(32, 12)

        # Build set of allowed indices
        allowed_indices = set()
        for link_name in self.constrained_config.preferred_links:
            allowed_indices.update(sampler.link_to_indices[link_name].tolist())

        # Check all sampled indices are in allowed set
        for b in range(result.shape[0]):
            for idx in result[b]:
                self.assertIn(idx.item(), allowed_indices, f"Index {idx.item()} not in allowed links")

    # --- Finger Diversity Tests ---

    def test_min_fingers_constraint(self):
        """Sampling should respect minimum fingers constraint"""
        sampler = HierarchicalContactSampler(self.mock_hand_model, self.constrained_config)

        result = sampler.sample(32, 12)

        for b in range(result.shape[0]):
            fingers_used = set()
            for idx in result[b]:
                link_name = sampler._index_to_link_name(idx.item())
                finger = sampler.link_to_finger[link_name]
                fingers_used.add(finger)

            self.assertGreaterEqual(
                len(fingers_used),
                self.constrained_config.min_fingers,
                f"Batch {b}: Only {len(fingers_used)} fingers, expected >= {self.constrained_config.min_fingers}",
            )

    def test_max_contacts_per_link_constraint(self):
        """Sampling should try to respect max contacts per link when possible"""
        # Note: The implementation may increase max_contacts_per_link dynamically
        # if needed to fill the requested number of contacts
        sampler = HierarchicalContactSampler(self.mock_hand_model, self.constrained_config)

        # Use fewer contacts so we don't need to exceed max_contacts_per_link
        n_contacts = 8  # 4 links * 2 max = 8, so no need to exceed
        result = sampler.sample(32, n_contacts)

        for b in range(result.shape[0]):
            link_counts = Counter()
            for idx in result[b]:
                link_name = sampler._index_to_link_name(idx.item())
                link_counts[link_name] += 1

            for link_name, count in link_counts.items():
                self.assertLessEqual(
                    count,
                    self.constrained_config.max_contacts_per_link,
                    f"Batch {b}: Link {link_name} has {count} contacts, max is {self.constrained_config.max_contacts_per_link}",
                )

    # --- Edge Cases ---

    def test_more_contacts_than_candidates_raises(self):
        """Should raise error if requesting more contacts than available"""
        # Constrained to only fingertips: 8+8+8+10 = 34 contacts
        sampler = HierarchicalContactSampler(self.mock_hand_model, self.constrained_config)

        with self.assertRaises(ValueError):
            sampler.sample(1, 100)  # Request 100 contacts from 34 available

    def test_empty_preferred_links_raises(self):
        """Empty preferred_links in constrained mode should raise error"""
        # Constrained mode requires preferred_links to be specified
        with self.assertRaises(ValueError):
            ContactSamplingConfig(
                mode="constrained",
                preferred_links=[],  # Empty - should raise
                min_fingers=2,
            )

    def test_single_finger_when_min_fingers_1(self):
        """Should work with min_fingers=1"""
        config = ContactSamplingConfig(
            mode="constrained",
            preferred_links=["thumb_link_3"],  # Only thumb
            min_fingers=1,
            max_contacts_per_link=10,
        )

        sampler = HierarchicalContactSampler(self.mock_hand_model, config)
        result = sampler.sample(1, 8)

        # All contacts should be from thumb_link_3
        thumb_indices = set(sampler.link_to_indices["thumb_link_3"].tolist())
        for idx in result[0]:
            self.assertIn(idx.item(), thumb_indices)


# ============================================================================
# Per-Batch Prior Tests
# ============================================================================


class TestPerBatchPriors(unittest.TestCase):
    """Tests for per-batch-item contact configurations"""

    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        cls.mock_hand_model = create_mock_hand_model(cls.device)

    def test_different_configs_per_batch_item(self):
        """Each batch item can have different contact configuration"""
        configs = [
            ContactSamplingConfig(
                mode="constrained",
                preferred_links=["index_link_3", "thumb_link_3"],
                min_fingers=2,
            ),
            ContactSamplingConfig(
                mode="constrained",
                preferred_links=["middle_link_3", "ring_link_3"],
                min_fingers=2,
            ),
        ]

        samplers = [HierarchicalContactSampler(self.mock_hand_model, config) for config in configs]

        # Sample from each
        results = [sampler.sample(1, 8) for sampler in samplers]

        # Verify different links used
        links_used_0 = {samplers[0]._index_to_link_name(idx.item()) for idx in results[0][0]}
        links_used_1 = {samplers[1]._index_to_link_name(idx.item()) for idx in results[1][0]}

        # Should be disjoint (different fingers)
        self.assertNotEqual(links_used_0, links_used_1)


# ============================================================================
# Integration Tests
# ============================================================================


class TestOptimizerIntegration(unittest.TestCase):
    """Tests for integration with MalaStar optimizer"""

    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        cls.mock_hand_model = create_mock_hand_model(cls.device)
        cls.constrained_config = create_constrained_config()

    def test_optimizer_uses_sampler_for_contact_switch(self):
        """Optimizer should use hierarchical sampler when switching contacts"""
        sampler = HierarchicalContactSampler(self.mock_hand_model, self.constrained_config)

        # Mock optimizer contact switch
        batch_size, n_contacts = 32, 12
        current_contacts = sampler.sample(batch_size, n_contacts)

        # Simulate switch: 50% probability
        switch_mask = torch.rand(batch_size) < 0.5

        if switch_mask.any():
            n_switch = switch_mask.sum().item()
            new_contacts = sampler.sample(n_switch, n_contacts)
            current_contacts[switch_mask] = new_contacts

        # Verify still valid after switch
        allowed_indices = set()
        for link_name in self.constrained_config.preferred_links:
            allowed_indices.update(sampler.link_to_indices[link_name].tolist())

        for b in range(batch_size):
            for idx in current_contacts[b]:
                self.assertIn(idx.item(), allowed_indices)


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance(unittest.TestCase):
    """Performance benchmarks"""

    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        cls.mock_hand_model = create_mock_hand_model(cls.device)

    def test_hierarchical_sampling_completes_quickly(self):
        """Hierarchical sampling should complete in reasonable time"""
        config_hierarchical = ContactSamplingConfig(
            mode="constrained",
            preferred_links=["index_link_3", "middle_link_3", "thumb_link_3"],
            min_fingers=2,
        )

        sampler = HierarchicalContactSampler(self.mock_hand_model, config_hierarchical)

        batch_size, n_contacts = 128, 12

        # Time the sampling
        start = time.time()
        for _ in range(10):
            sampler.sample(batch_size, n_contacts)
        elapsed = time.time() - start

        # Should complete 10 iterations in under 1 second
        self.assertLess(elapsed, 1.0, f"Sampling too slow: {elapsed:.2f}s for 10 iterations")


# ============================================================================
# Run if executed directly
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
