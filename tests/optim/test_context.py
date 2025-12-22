"""
Unit tests for OptimizationContext.
"""

import unittest
from unittest.mock import MagicMock

import torch

from graspqp.optim.context import OptimizationContext
from graspqp.optim.state import ReferenceTrajectory


class TestOptimizationContext(unittest.TestCase):
    """Tests for OptimizationContext."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hand_model = MagicMock()
        self.object_model = MagicMock()
        self.device = torch.device("cpu")
        
        # Create a mock reference
        self.B = 4
        self.T = 10
        self.D_hand = 25
        self.D_obj = 7
        self.n_contacts = 8
        
        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)
        self.reference = ReferenceTrajectory(
            hand_states=hand,
            object_states=obj,
            contact_fingers=["thumb", "index"],
            n_contacts=self.n_contacts,
        )
        
    def test_basic_creation(self):
        """Test basic context creation."""
        ctx = OptimizationContext(
            self.hand_model,
            self.object_model,
            reference=self.reference,
        )
        
        self.assertIs(ctx.hand_model, self.hand_model)
        self.assertIs(ctx.object_model, self.object_model)
        self.assertIs(ctx.reference, self.reference)
        
    def test_contact_config_from_reference(self):
        """Test that contact configuration is set from reference."""
        ctx = OptimizationContext(
            self.hand_model,
            self.object_model,
            reference=self.reference,
        )
        
        self.assertEqual(ctx.contact_fingers, self.reference.contact_fingers)
        self.assertEqual(ctx.n_contacts, self.reference.n_contacts)
        
    def test_set_contact_indices(self):
        """Test manually setting contact indices (by optimizer)."""
        ctx = OptimizationContext(self.hand_model, self.object_model)
        
        # Initially no contact indices (optimizer hasn't set them yet)
        self.assertIsNone(ctx.contact_indices)
        
        # Optimizer sets indices during step
        indices = torch.randint(0, 100, (self.B, self.n_contacts))
        ctx.set_contact_indices(indices)
        
        self.assertTrue(torch.equal(ctx.contact_indices, indices))
        
    def test_cache_set_and_get(self):
        """Test cache set and get."""
        ctx = OptimizationContext(self.hand_model, self.object_model)
        
        value = torch.randn(10, 10)
        ctx.set_cached("test_key", value, scope="step")
        
        retrieved = ctx.get_cached("test_key")
        self.assertTrue(torch.equal(retrieved, value))
        
    def test_cache_get_nonexistent(self):
        """Test getting a non-existent cache key."""
        ctx = OptimizationContext(self.hand_model, self.object_model)
        
        result = ctx.get_cached("nonexistent")
        self.assertIsNone(result)
        
    def test_get_or_compute(self):
        """Test lazy computation with cache."""
        ctx = OptimizationContext(self.hand_model, self.object_model)
        
        call_count = [0]
        
        def compute_fn():
            call_count[0] += 1
            return torch.randn(10)
        
        # First call should compute
        result1 = ctx.get_or_compute("test_key", compute_fn)
        self.assertEqual(call_count[0], 1)
        
        # Second call should use cache
        result2 = ctx.get_or_compute("test_key", compute_fn)
        self.assertEqual(call_count[0], 1)  # Not called again
        self.assertTrue(torch.equal(result1, result2))
        
    def test_clear_step_cache(self):
        """Test clearing step cache."""
        ctx = OptimizationContext(self.hand_model, self.object_model)
        
        ctx.set_cached("step_value", torch.randn(10), scope="step")
        ctx.set_cached("trajectory_value", torch.randn(10), scope="trajectory")
        
        ctx.clear_step_cache()
        
        self.assertIsNone(ctx.get_cached("step_value"))
        self.assertIsNotNone(ctx.get_cached("trajectory_value"))
        
    def test_clear_all_cache(self):
        """Test clearing all cache."""
        ctx = OptimizationContext(self.hand_model, self.object_model)
        
        ctx.set_cached("step_value", torch.randn(10), scope="step")
        ctx.set_cached("trajectory_value", torch.randn(10), scope="trajectory")
        
        ctx.clear_all_cache()
        
        self.assertIsNone(ctx.get_cached("step_value"))
        self.assertIsNone(ctx.get_cached("trajectory_value"))


if __name__ == "__main__":
    unittest.main()

