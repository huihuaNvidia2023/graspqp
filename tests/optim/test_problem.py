"""
Unit tests for OptimizationProblem.
"""

import unittest
from unittest.mock import MagicMock

import torch

from graspqp.optim.context import OptimizationContext
from graspqp.optim.costs import ReferenceTrackingCost, VelocitySmoothnessCost
from graspqp.optim.problem import OptimizationProblem
from graspqp.optim.state import ReferenceTrajectory, TrajectoryState


class TestOptimizationProblem(unittest.TestCase):
    """Tests for OptimizationProblem."""

    def setUp(self):
        self.B = 4
        self.T = 10
        self.D_hand = 25
        self.D_obj = 7
        self.n_contacts = 8

        # Create reference
        ref_hand = torch.randn(self.B, self.T, self.D_hand)
        ref_obj = torch.randn(self.B, self.T, self.D_obj)
        self.reference = ReferenceTrajectory(
            hand_states=ref_hand,
            object_states=ref_obj,
            contact_fingers=["thumb", "index"],
            n_contacts=self.n_contacts,
        )

        self.ctx = OptimizationContext(
            hand_model=MagicMock(),
            object_model=MagicMock(),
            reference=self.reference,
        )

    def test_add_cost(self):
        """Test adding costs to problem."""
        problem = OptimizationProblem(self.ctx)

        cost1 = ReferenceTrackingCost(name="ref", weight=100.0)
        cost2 = VelocitySmoothnessCost(name="vel", weight=1.0)

        problem.add_cost(cost1)
        problem.add_cost(cost2)

        self.assertEqual(len(problem.costs), 2)
        self.assertIn("ref", problem.costs)
        self.assertIn("vel", problem.costs)

    def test_add_duplicate_name_raises(self):
        """Test that adding duplicate cost name raises."""
        problem = OptimizationProblem(self.ctx)

        cost1 = ReferenceTrackingCost(name="ref", weight=100.0)
        cost2 = ReferenceTrackingCost(name="ref", weight=50.0)

        problem.add_cost(cost1)

        with self.assertRaises(ValueError):
            problem.add_cost(cost2)

    def test_remove_cost(self):
        """Test removing costs."""
        problem = OptimizationProblem(self.ctx)

        cost = ReferenceTrackingCost(name="ref", weight=100.0)
        problem.add_cost(cost)

        problem.remove_cost("ref")

        self.assertEqual(len(problem.costs), 0)

    def test_set_weight(self):
        """Test setting cost weight."""
        problem = OptimizationProblem(self.ctx)

        cost = ReferenceTrackingCost(name="ref", weight=100.0)
        problem.add_cost(cost)

        problem.set_weight("ref", 50.0)

        self.assertEqual(problem.costs["ref"].weight, 50.0)

    def test_set_enabled(self):
        """Test enabling/disabling costs."""
        problem = OptimizationProblem(self.ctx)

        cost = ReferenceTrackingCost(name="ref", weight=100.0)
        problem.add_cost(cost)

        problem.set_enabled("ref", False)

        self.assertFalse(problem.costs["ref"].enabled)

    def test_total_energy(self):
        """Test total energy computation."""
        problem = OptimizationProblem(self.ctx)

        cost = VelocitySmoothnessCost(name="vel", weight=1.0)
        problem.add_cost(cost)

        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)
        state = TrajectoryState(hand, obj)

        energy = problem.total_energy(state)

        self.assertEqual(energy.shape, (self.B,))
        self.assertTrue((energy >= 0).all())

    def test_evaluate_all(self):
        """Test evaluating all costs."""
        problem = OptimizationProblem(self.ctx)

        problem.add_cost(VelocitySmoothnessCost(name="vel", weight=1.0))
        problem.add_cost(ReferenceTrackingCost(name="ref", weight=10.0))

        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)
        state = TrajectoryState(hand, obj)

        results = problem.evaluate_all(state)

        self.assertIn("vel", results)
        self.assertIn("ref", results)
        self.assertEqual(results["vel"].shape, (self.B,))
        self.assertEqual(results["ref"].shape, (self.B,))

    def test_cost_breakdown(self):
        """Test cost breakdown for logging."""
        problem = OptimizationProblem(self.ctx)

        problem.add_cost(VelocitySmoothnessCost(name="vel", weight=1.0))

        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)
        state = TrajectoryState(hand, obj)

        breakdown = problem.cost_breakdown(state)

        self.assertIn("vel", breakdown)
        self.assertIsInstance(breakdown["vel"], float)

    def test_from_dict(self):
        """Test creating problem from config dict."""
        config = {
            "costs": [
                {
                    "type": "VelocitySmoothnessCost",
                    "name": "vel",
                    "weight": 1.0,
                },
                {
                    "type": "ReferenceTrackingCost",
                    "name": "ref",
                    "weight": 100.0,
                },
            ]
        }

        problem = OptimizationProblem.from_dict(config, self.ctx)

        self.assertEqual(len(problem.costs), 2)
        self.assertIn("vel", problem.costs)
        self.assertIn("ref", problem.costs)


class TestOptimizationProblemWithPerturbations(unittest.TestCase):
    """Tests for problem with multi-perturbation states."""

    def setUp(self):
        self.B = 4
        self.K = 8
        self.T = 10
        self.D_hand = 25
        self.D_obj = 7

        self.ctx = OptimizationContext(
            hand_model=MagicMock(),
            object_model=MagicMock(),
        )

    def test_total_energy_with_perturbations(self):
        """Test total energy with (B, K, T, D) state."""
        problem = OptimizationProblem(self.ctx)
        problem.add_cost(VelocitySmoothnessCost(name="vel", weight=1.0))

        hand = torch.randn(self.B, self.K, self.T, self.D_hand)
        obj = torch.randn(self.B, self.K, self.T, self.D_obj)
        state = TrajectoryState(hand, obj)

        energy = problem.total_energy(state)

        # Should return B*K energies
        self.assertEqual(energy.shape, (self.B * self.K,))


if __name__ == "__main__":
    unittest.main()
