"""
Unit tests for grasp-specific cost functions.
"""

import unittest
from unittest.mock import MagicMock

import torch

from graspqp.optim.context import OptimizationContext
from graspqp.optim.costs import ContactDistanceCost, ForceClosureCost, JointLimitCost, PriorPoseCost
from graspqp.optim.state import TrajectoryState


class MockHandModel:
    """Mock hand model for testing."""

    def __init__(self, batch_size, n_contacts=8, device="cpu"):
        self.batch_size = batch_size
        self.n_contacts = n_contacts
        self.device = device
        self._hand_pose = None
        self._contact_points = None

    @property
    def hand_pose(self):
        return self._hand_pose

    @hand_pose.setter
    def hand_pose(self, value):
        self._hand_pose = value

    @property
    def contact_points(self):
        if self._contact_points is None:
            B = self._hand_pose.shape[0]
            self._contact_points = torch.randn(B, self.n_contacts, 3, device=self.device)
        return self._contact_points

    def set_parameters(self, params):
        self._hand_pose = params
        self._contact_points = None  # Reset

    def get_joint_limits_violations(self):
        B = self._hand_pose.shape[0]
        return torch.rand(B, device=self.device) * 0.1


class MockObjectModel:
    """Mock object model for testing."""

    def __init__(self, device="cpu"):
        self.device = device
        self.cog = torch.zeros(3, device=device)

    def cal_distance(self, points):
        """Return mock SDF values and normals."""
        B, N, _ = points.shape
        distances = torch.rand(B, N, device=self.device) * 0.05 - 0.02  # Small distances
        normals = torch.randn(B, N, 3, device=self.device)
        normals = normals / normals.norm(dim=-1, keepdim=True)
        return distances, normals


class TestContactDistanceCost(unittest.TestCase):
    """Tests for ContactDistanceCost."""

    def setUp(self):
        self.B = 4
        self.T = 1  # Single frame
        self.D_hand = 25
        self.device = "cpu"

        self.hand_model = MockHandModel(self.B, device=self.device)
        self.object_model = MockObjectModel(device=self.device)

        self.ctx = OptimizationContext(
            hand_model=self.hand_model,
            object_model=self.object_model,
            device=self.device,
        )

    def test_basic_evaluation(self):
        """Test basic cost evaluation."""
        cost = ContactDistanceCost(weight=100.0)

        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, 7)
        state = TrajectoryState(hand, obj)

        result = cost.evaluate(state, self.ctx)

        self.assertEqual(result.shape, (self.B,))
        self.assertTrue((result >= 0).all())


class TestForceClosureCost(unittest.TestCase):
    """Tests for ForceClosureCost."""

    def setUp(self):
        self.B = 4
        self.T = 1
        self.D_hand = 25
        self.device = "cpu"

        self.hand_model = MockHandModel(self.B, device=self.device)
        self.object_model = MockObjectModel(device=self.device)

        self.ctx = OptimizationContext(
            hand_model=self.hand_model,
            object_model=self.object_model,
            device=self.device,
        )

    def test_without_energy_function(self):
        """Test that cost returns zeros without energy function."""
        cost = ForceClosureCost(weight=1.0)

        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, 7)
        state = TrajectoryState(hand, obj)

        result = cost.evaluate(state, self.ctx)

        self.assertEqual(result.shape, (self.B,))
        self.assertTrue(torch.allclose(result, torch.zeros(self.B)))

    def test_with_mock_energy_function(self):
        """Test with a mock energy function."""
        cost = ForceClosureCost(weight=1.0)

        # Mock energy function
        def mock_energy_fnc(contact_pts, contact_normals, sdf, cog, with_solution, svd_gain):
            B = contact_pts.shape[0]
            return torch.rand(B), None

        cost.set_energy_function(mock_energy_fnc)

        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, 7)
        state = TrajectoryState(hand, obj)

        result = cost.evaluate(state, self.ctx)

        self.assertEqual(result.shape, (self.B,))


class TestJointLimitCost(unittest.TestCase):
    """Tests for JointLimitCost."""

    def setUp(self):
        self.B = 4
        self.T = 1
        self.D_hand = 25
        self.device = "cpu"

        self.hand_model = MockHandModel(self.B, device=self.device)
        self.object_model = MockObjectModel(device=self.device)

        self.ctx = OptimizationContext(
            hand_model=self.hand_model,
            object_model=self.object_model,
            device=self.device,
        )

    def test_basic_evaluation(self):
        """Test basic cost evaluation."""
        cost = JointLimitCost(weight=1.0)

        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, 7)
        state = TrajectoryState(hand, obj)

        result = cost.evaluate(state, self.ctx)

        self.assertEqual(result.shape, (self.B,))
        self.assertTrue((result >= 0).all())


class TestPriorPoseCost(unittest.TestCase):
    """Tests for PriorPoseCost."""

    def setUp(self):
        self.B = 4
        self.T = 1
        self.D_hand = 25
        self.device = "cpu"

        self.ctx = MagicMock()

    def test_without_prior(self):
        """Test that cost returns zeros without prior set."""
        cost = PriorPoseCost(weight=10.0)

        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, 7)
        state = TrajectoryState(hand, obj)

        result = cost.evaluate(state, self.ctx)

        self.assertEqual(result.shape, (self.B,))
        self.assertTrue(torch.allclose(result, torch.zeros(self.B)))

    def test_with_prior(self):
        """Test with prior pose set."""
        cost = PriorPoseCost(weight=10.0)

        # Set prior
        prior = torch.randn(self.B, self.D_hand)
        cost.set_prior_pose(prior)

        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, 7)
        state = TrajectoryState(hand, obj)

        result = cost.evaluate(state, self.ctx)

        self.assertEqual(result.shape, (self.B,))
        self.assertTrue((result > 0).all())  # Should be positive (L2 distance)

    def test_zero_cost_when_matching_prior(self):
        """Test that cost is zero when state matches prior."""
        cost = PriorPoseCost(weight=10.0)

        # Set prior same as state
        prior = torch.randn(self.B, self.D_hand)
        cost.set_prior_pose(prior)

        hand = prior.unsqueeze(1)  # (B, 1, D)
        obj = torch.randn(self.B, self.T, 7)
        state = TrajectoryState(hand, obj)

        result = cost.evaluate(state, self.ctx)

        self.assertTrue(torch.allclose(result, torch.zeros(self.B), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
