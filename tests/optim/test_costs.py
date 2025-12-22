"""
Unit tests for cost functions.
"""

import unittest
from unittest.mock import MagicMock

import torch

from graspqp.optim.context import OptimizationContext
from graspqp.optim.costs import (AccelerationCost, CostFunction, PerFrameCost, ReferenceTrackingCost, TemporalCost,
                                 VelocitySmoothnessCost)
from graspqp.optim.costs.registry import CostRegistry, create_cost
from graspqp.optim.state import ReferenceTrajectory, TrajectoryState


class DummyPerFrameCost(PerFrameCost):
    """Dummy per-frame cost for testing."""

    def evaluate_frames(self, state, ctx):
        # Return sum of squared hand states per frame
        return (state.hand_states**2).sum(dim=-1)  # (B, T)


class TestCostFunction(unittest.TestCase):
    """Tests for CostFunction base class."""

    def test_disabled_cost_returns_zeros(self):
        """Test that disabled costs return zeros."""
        cost = DummyPerFrameCost(name="dummy", weight=1.0, enabled=False)

        B, T, D = 4, 10, 25
        hand = torch.randn(B, T, D)
        obj = torch.randn(B, T, 7)
        state = TrajectoryState(hand, obj)
        ctx = MagicMock()

        result = cost(state, ctx)

        self.assertEqual(result.shape, (B,))
        self.assertTrue((result == 0).all())

    def test_weight_applied(self):
        """Test that weight is applied to cost."""
        cost1 = DummyPerFrameCost(name="dummy1", weight=1.0)
        cost2 = DummyPerFrameCost(name="dummy2", weight=10.0)

        B, T, D = 4, 10, 25
        hand = torch.randn(B, T, D)
        obj = torch.randn(B, T, 7)
        state = TrajectoryState(hand, obj)
        ctx = MagicMock()

        result1 = cost1(state, ctx)
        result2 = cost2(state, ctx)

        # Result2 should be 10x result1
        self.assertTrue(torch.allclose(result2, result1 * 10))


class TestPerFrameCost(unittest.TestCase):
    """Tests for PerFrameCost base class."""

    def test_aggregation_sum(self):
        """Test sum aggregation."""
        cost = DummyPerFrameCost(name="dummy", aggregation="sum")

        B, T, D = 4, 10, 25
        hand = torch.ones(B, T, D)  # Use ones for predictable result
        obj = torch.randn(B, T, 7)
        state = TrajectoryState(hand, obj)
        ctx = MagicMock()

        result = cost.evaluate(state, ctx)

        expected = T * D  # Sum of D ones per frame, T frames
        self.assertTrue(torch.allclose(result, torch.full((B,), float(expected))))

    def test_aggregation_mean(self):
        """Test mean aggregation."""
        cost = DummyPerFrameCost(name="dummy", aggregation="mean")

        B, T, D = 4, 10, 25
        hand = torch.ones(B, T, D)
        obj = torch.randn(B, T, 7)
        state = TrajectoryState(hand, obj)
        ctx = MagicMock()

        result = cost.evaluate(state, ctx)

        expected = D  # Mean per frame is D, and we average over frames
        self.assertTrue(torch.allclose(result, torch.full((B,), float(expected))))

    def test_with_valid_mask(self):
        """Test that valid mask is applied."""
        cost = DummyPerFrameCost(name="dummy", aggregation="sum")

        B, T, D = 4, 10, 25
        hand = torch.ones(B, T, D)
        obj = torch.randn(B, T, 7)
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[0, 5:] = False  # First batch has only 5 valid frames

        state = TrajectoryState(hand, obj, valid_mask=mask)
        ctx = MagicMock()

        result = cost.evaluate(state, ctx)

        # First batch should have half the cost
        expected_full = T * D
        expected_partial = 5 * D
        self.assertAlmostEqual(result[0].item(), expected_partial, places=5)
        self.assertAlmostEqual(result[1].item(), expected_full, places=5)


class TestReferenceTrackingCost(unittest.TestCase):
    """Tests for ReferenceTrackingCost."""

    def setUp(self):
        self.B = 4
        self.T = 10
        self.D_hand = 25
        self.D_obj = 7
        self.n_contacts = 8

        # Create reference
        self.ref_hand = torch.randn(self.B, self.T, self.D_hand)
        self.ref_obj = torch.randn(self.B, self.T, self.D_obj)
        contacts = torch.randint(0, 100, (self.B, self.n_contacts))

        self.reference = ReferenceTrajectory(self.ref_hand, self.ref_obj, contacts)

        self.ctx = OptimizationContext(
            hand_model=MagicMock(),
            object_model=MagicMock(),
            reference=self.reference,
        )

    def test_zero_cost_when_matching_reference(self):
        """Test that cost is zero when state matches reference."""
        cost = ReferenceTrackingCost(weight=1.0)

        # State equals reference
        state = TrajectoryState(
            self.ref_hand.clone(),
            self.ref_obj.clone(),
        )

        result = cost.evaluate(state, self.ctx)

        self.assertTrue(torch.allclose(result, torch.zeros(self.B), atol=1e-6))

    def test_positive_cost_when_different(self):
        """Test that cost is positive when state differs from reference."""
        cost = ReferenceTrackingCost(weight=1.0)

        # State differs from reference
        state = TrajectoryState(
            self.ref_hand + 0.1,
            self.ref_obj + 0.1,
        )

        result = cost.evaluate(state, self.ctx)

        self.assertTrue((result > 0).all())


class TestVelocitySmoothnessCost(unittest.TestCase):
    """Tests for VelocitySmoothnessCost."""

    def setUp(self):
        self.B = 4
        self.T = 10
        self.D_hand = 25
        self.D_obj = 7

        self.ctx = MagicMock()

    def test_zero_cost_for_constant_trajectory(self):
        """Test that constant trajectory has zero velocity cost."""
        cost = VelocitySmoothnessCost(weight=1.0)

        # Constant trajectory
        hand = torch.ones(self.B, self.T, self.D_hand)
        obj = torch.ones(self.B, self.T, self.D_obj)
        state = TrajectoryState(hand, obj)

        result = cost.evaluate(state, self.ctx)

        self.assertTrue(torch.allclose(result, torch.zeros(self.B), atol=1e-6))

    def test_positive_cost_for_varying_trajectory(self):
        """Test that varying trajectory has positive velocity cost."""
        cost = VelocitySmoothnessCost(weight=1.0)

        # Varying trajectory
        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)
        state = TrajectoryState(hand, obj)

        result = cost.evaluate(state, self.ctx)

        self.assertTrue((result > 0).all())

    def test_component_selection(self):
        """Test selecting hand-only or object-only component."""
        # Hand-only
        cost_hand = VelocitySmoothnessCost(weight=1.0, config={"component": "hand"})
        # Object-only
        cost_obj = VelocitySmoothnessCost(weight=1.0, config={"component": "object"})
        # Both
        cost_both = VelocitySmoothnessCost(weight=1.0, config={"component": "both"})

        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)
        state = TrajectoryState(hand, obj)

        result_hand = cost_hand.evaluate(state, self.ctx)
        result_obj = cost_obj.evaluate(state, self.ctx)
        result_both = cost_both.evaluate(state, self.ctx)

        # Both should be approximately hand + object
        self.assertTrue(torch.allclose(result_both, result_hand + result_obj, rtol=1e-5))


class TestAccelerationCost(unittest.TestCase):
    """Tests for AccelerationCost."""

    def setUp(self):
        self.B = 4
        self.T = 10
        self.D_hand = 25
        self.D_obj = 7
        self.ctx = MagicMock()

    def test_zero_cost_for_linear_trajectory(self):
        """Test that linear trajectory has zero acceleration cost."""
        cost = AccelerationCost(weight=1.0)

        # Linear trajectory: x(t) = t
        t = torch.arange(self.T).float().view(1, self.T, 1)
        hand = t.expand(self.B, self.T, self.D_hand)
        obj = t.expand(self.B, self.T, self.D_obj)
        state = TrajectoryState(hand, obj)

        result = cost.evaluate(state, self.ctx)

        self.assertTrue(torch.allclose(result, torch.zeros(self.B), atol=1e-4))

    def test_short_trajectory(self):
        """Test that short trajectories (T<3) return zero."""
        cost = AccelerationCost(weight=1.0)

        # Only 2 frames
        hand = torch.randn(self.B, 2, self.D_hand)
        obj = torch.randn(self.B, 2, self.D_obj)
        state = TrajectoryState(hand, obj)

        result = cost.evaluate(state, self.ctx)

        self.assertTrue(torch.allclose(result, torch.zeros(self.B)))


class TestCostRegistry(unittest.TestCase):
    """Tests for CostRegistry."""

    def test_list_available(self):
        """Test listing available costs."""
        available = CostRegistry.list_available()

        self.assertIn("ReferenceTrackingCost", available)
        self.assertIn("PenetrationCost", available)
        self.assertIn("VelocitySmoothnessCost", available)

    def test_create_cost(self):
        """Test creating cost from config."""
        config = {
            "type": "ReferenceTrackingCost",
            "name": "ref_track",
            "weight": 50.0,
            "config": {"hand_weight": 2.0},
        }

        cost = create_cost(config)

        self.assertIsInstance(cost, ReferenceTrackingCost)
        self.assertEqual(cost.name, "ref_track")
        self.assertEqual(cost.weight, 50.0)
        self.assertEqual(cost.hand_weight, 2.0)

    def test_unknown_cost_raises(self):
        """Test that unknown cost type raises error."""
        with self.assertRaises(ValueError):
            CostRegistry.get("NonExistentCost")


if __name__ == "__main__":
    unittest.main()
