"""
Unit tests for optimizers.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from graspqp.optim.context import OptimizationContext
from graspqp.optim.optimizers import AdamOptimizer, SGDOptimizer
from graspqp.optim.problem import OptimizationProblem
from graspqp.optim.state import TrajectoryState


class DummyProblem:
    """Mock optimization problem for testing."""

    def __init__(self):
        self.context = MagicMock()
        self.context.clear_step_cache = MagicMock()

    def total_energy(self, state):
        # Simple quadratic energy: ||x||^2
        # Flatten all spatial/temporal dims for both 3D and 4D cases
        hand_flat = state.hand_states.reshape(state.hand_states.shape[0], -1)
        obj_flat = state.object_states.reshape(state.object_states.shape[0], -1)
        energy = (hand_flat**2).sum(dim=-1) + (obj_flat**2).sum(dim=-1)
        return energy


class TestAdamOptimizer(unittest.TestCase):
    """Tests for AdamOptimizer."""

    def setUp(self):
        self.B = 4
        self.T = 10
        self.D_hand = 25
        self.D_obj = 7

    def test_basic_step(self):
        """Test that Adam step reduces energy."""
        optimizer = AdamOptimizer(lr=0.1)
        problem = DummyProblem()

        # Start with non-zero state
        hand = torch.ones(self.B, self.T, self.D_hand)
        obj = torch.ones(self.B, self.T, self.D_obj)
        state = TrajectoryState(hand, obj)

        initial_energy = problem.total_energy(state)

        # Take a step
        new_state = optimizer.step(state, problem)

        new_energy = problem.total_energy(new_state)

        # Energy should decrease (moving toward zero)
        self.assertTrue((new_energy < initial_energy).all())

    def test_step_count_increments(self):
        """Test that step count increments."""
        optimizer = AdamOptimizer()
        problem = DummyProblem()

        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)
        state = TrajectoryState(hand, obj)

        self.assertEqual(optimizer.step_count, 0)

        optimizer.step(state, problem)
        self.assertEqual(optimizer.step_count, 1)

        optimizer.step(state, problem)
        self.assertEqual(optimizer.step_count, 2)

    def test_reset(self):
        """Test optimizer reset."""
        optimizer = AdamOptimizer()
        problem = DummyProblem()

        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)
        state = TrajectoryState(hand, obj)

        optimizer.step(state, problem)
        self.assertEqual(optimizer.step_count, 1)

        optimizer.reset()
        self.assertEqual(optimizer.step_count, 0)

    def test_diagnostics(self):
        """Test getting diagnostics."""
        optimizer = AdamOptimizer(lr=0.05)

        diag = optimizer.get_diagnostics()

        self.assertEqual(diag["type"], "AdamOptimizer")
        self.assertEqual(diag["lr"], 0.05)
        self.assertEqual(diag["step_count"], 0)


class TestSGDOptimizer(unittest.TestCase):
    """Tests for SGDOptimizer."""

    def setUp(self):
        self.B = 4
        self.T = 10
        self.D_hand = 25
        self.D_obj = 7

    def test_basic_step(self):
        """Test that SGD step works."""
        optimizer = SGDOptimizer(lr=0.1, momentum=0.9)
        problem = DummyProblem()

        hand = torch.ones(self.B, self.T, self.D_hand)
        obj = torch.ones(self.B, self.T, self.D_obj)
        state = TrajectoryState(hand, obj)

        initial_energy = problem.total_energy(state)
        new_state = optimizer.step(state, problem)
        new_energy = problem.total_energy(new_state)

        # Energy should decrease
        self.assertTrue((new_energy < initial_energy).all())


class TestOptimizerWithPerturbations(unittest.TestCase):
    """Tests for optimizers with multi-perturbation states."""

    def setUp(self):
        self.B = 4
        self.K = 8
        self.T = 10
        self.D_hand = 25
        self.D_obj = 7

    def test_adam_with_perturbations(self):
        """Test Adam with (B, K, T, D) state."""
        optimizer = AdamOptimizer(lr=0.1)
        problem = DummyProblem()

        hand = torch.ones(self.B, self.K, self.T, self.D_hand)
        obj = torch.ones(self.B, self.K, self.T, self.D_obj)
        state = TrajectoryState(hand, obj)

        self.assertTrue(state._has_perturbations)

        new_state = optimizer.step(state, problem)

        # State should still have perturbations
        self.assertEqual(new_state.B, self.B)
        self.assertEqual(new_state.K, self.K)


if __name__ == "__main__":
    unittest.main()
