"""
Unit tests for TrajectoryState and ReferenceTrajectory.
"""

import unittest

import torch

from graspqp.optim.state import ReferenceTrajectory, ResultSelector, TrajectoryState


class TestTrajectoryState(unittest.TestCase):
    """Tests for TrajectoryState."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.B = 4  # batch size
        self.T = 10  # trajectory length
        self.D_hand = 25  # hand state dimension
        self.D_obj = 7  # object state dimension

    def test_basic_creation(self):
        """Test basic TrajectoryState creation."""
        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)

        state = TrajectoryState(hand, obj)

        self.assertEqual(state.B, self.B)
        self.assertEqual(state.T, self.T)
        self.assertEqual(state.D_hand, self.D_hand)
        self.assertEqual(state.D_obj, self.D_obj)
        self.assertEqual(state.K, 1)
        self.assertFalse(state._has_perturbations)

    def test_with_perturbations(self):
        """Test TrajectoryState with multiple perturbations."""
        K = 8
        hand = torch.randn(self.B, K, self.T, self.D_hand)
        obj = torch.randn(self.B, K, self.T, self.D_obj)

        state = TrajectoryState(hand, obj)

        self.assertEqual(state.B, self.B)
        self.assertEqual(state.K, K)
        self.assertEqual(state.T, self.T)
        self.assertTrue(state._has_perturbations)

    def test_default_valid_mask(self):
        """Test that valid_mask is created by default."""
        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)

        state = TrajectoryState(hand, obj)

        self.assertIsNotNone(state.valid_mask)
        self.assertEqual(state.valid_mask.shape, (self.B, self.T))
        self.assertTrue(state.valid_mask.all())

    def test_flat_hand(self):
        """Test flattening hand states for batched FK."""
        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)

        state = TrajectoryState(hand, obj)
        flat = state.flat_hand

        self.assertEqual(flat.shape, (self.B * self.T, self.D_hand))

    def test_flat_hand_with_perturbations(self):
        """Test flattening with perturbations."""
        K = 8
        hand = torch.randn(self.B, K, self.T, self.D_hand)
        obj = torch.randn(self.B, K, self.T, self.D_obj)

        state = TrajectoryState(hand, obj)
        flat = state.flat_hand

        self.assertEqual(flat.shape, (self.B * K * self.T, self.D_hand))

    def test_get_frame(self):
        """Test getting a specific frame."""
        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)

        state = TrajectoryState(hand, obj)
        h, o = state.get_frame(5)

        self.assertEqual(h.shape, (self.B, self.D_hand))
        self.assertEqual(o.shape, (self.B, self.D_obj))
        self.assertTrue(torch.allclose(h, hand[:, 5]))
        self.assertTrue(torch.allclose(o, obj[:, 5]))

    def test_velocities(self):
        """Test velocity computation."""
        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)
        dt = 0.1

        state = TrajectoryState(hand, obj, dt=dt)
        vel = state.velocities("hand")

        self.assertEqual(vel.shape, (self.B, self.T - 1, self.D_hand))

        # Check velocity is correctly computed
        expected = (hand[:, 1:] - hand[:, :-1]) / dt
        self.assertTrue(torch.allclose(vel, expected))

    def test_accelerations(self):
        """Test acceleration computation."""
        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)

        state = TrajectoryState(hand, obj)
        acc = state.accelerations("hand")

        self.assertEqual(acc.shape, (self.B, self.T - 2, self.D_hand))

    def test_flatten_perturbations(self):
        """Test flattening perturbations."""
        K = 8
        hand = torch.randn(self.B, K, self.T, self.D_hand)
        obj = torch.randn(self.B, K, self.T, self.D_obj)

        state = TrajectoryState(hand, obj)
        flat_state = state.flatten_perturbations()

        self.assertEqual(flat_state.B, self.B * K)
        self.assertEqual(flat_state.T, self.T)
        self.assertFalse(flat_state._has_perturbations)

    def test_unflatten_perturbations(self):
        """Test unflattening perturbations."""
        K = 8
        hand = torch.randn(self.B * K, self.T, self.D_hand)
        obj = torch.randn(self.B * K, self.T, self.D_obj)

        state = TrajectoryState(hand, obj)
        unflat_state = state.unflatten_perturbations(self.B, K)

        self.assertEqual(unflat_state.B, self.B)
        self.assertEqual(unflat_state.K, K)
        self.assertTrue(unflat_state._has_perturbations)

    def test_clone(self):
        """Test cloning."""
        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)

        state = TrajectoryState(hand, obj)
        cloned = state.clone()

        self.assertTrue(torch.allclose(state.hand_states, cloned.hand_states))
        self.assertFalse(state.hand_states is cloned.hand_states)

    def test_detach(self):
        """Test detaching from computation graph."""
        hand = torch.randn(self.B, self.T, self.D_hand, requires_grad=True)
        obj = torch.randn(self.B, self.T, self.D_obj, requires_grad=True)

        state = TrajectoryState(hand, obj)
        detached = state.detach()

        self.assertFalse(detached.hand_states.requires_grad)

    def test_requires_grad(self):
        """Test setting requires_grad."""
        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)

        state = TrajectoryState(hand, obj)
        state.requires_grad_(True)

        self.assertTrue(state.hand_states.requires_grad)
        self.assertTrue(state.object_states.requires_grad)

    def test_lengths(self):
        """Test computing trajectory lengths from mask."""
        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)
        mask = torch.ones(self.B, self.T, dtype=torch.bool)
        mask[0, 5:] = False  # First trajectory has only 5 valid frames
        mask[1, 8:] = False  # Second has 8

        state = TrajectoryState(hand, obj, valid_mask=mask)
        lengths = state.lengths

        self.assertEqual(lengths[0].item(), 5)
        self.assertEqual(lengths[1].item(), 8)
        self.assertEqual(lengths[2].item(), self.T)
        self.assertEqual(lengths[3].item(), self.T)


class TestReferenceTrajectory(unittest.TestCase):
    """Tests for ReferenceTrajectory."""

    def setUp(self):
        self.B = 4
        self.T = 10
        self.D_hand = 25
        self.D_obj = 7
        self.n_contacts = 8

    def test_basic_creation(self):
        """Test basic ReferenceTrajectory creation."""
        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)
        contacts = torch.randint(0, 100, (self.B, self.n_contacts))

        ref = ReferenceTrajectory(hand, obj, contacts)

        self.assertEqual(ref.B, self.B)
        self.assertEqual(ref.T, self.T)
        self.assertEqual(ref.n_contacts, self.n_contacts)

    def test_to_device(self):
        """Test moving to device."""
        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)
        contacts = torch.randint(0, 100, (self.B, self.n_contacts))

        ref = ReferenceTrajectory(hand, obj, contacts)
        moved = ref.to(torch.device("cpu"))

        self.assertEqual(moved.device, torch.device("cpu"))


class TestTrajectoryStateFromReference(unittest.TestCase):
    """Tests for creating TrajectoryState from ReferenceTrajectory."""

    def setUp(self):
        self.B = 4
        self.T = 10
        self.D_hand = 25
        self.D_obj = 7
        self.n_contacts = 8

    def test_single_perturbation(self):
        """Test creating state with single perturbation (K=1)."""
        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)
        contacts = torch.randint(0, 100, (self.B, self.n_contacts))

        ref = ReferenceTrajectory(hand, obj, contacts)
        state = TrajectoryState.from_reference(ref, n_perturbations=1)

        self.assertEqual(state.B, self.B)
        self.assertEqual(state.K, 1)
        self.assertFalse(state._has_perturbations)
        self.assertTrue(torch.allclose(state.hand_states, hand))

    def test_multiple_perturbations(self):
        """Test creating state with multiple perturbations."""
        K = 8
        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)
        contacts = torch.randint(0, 100, (self.B, self.n_contacts))

        ref = ReferenceTrajectory(hand, obj, contacts)
        state = TrajectoryState.from_reference(ref, n_perturbations=K, perturbation_scale=0.01)

        self.assertEqual(state.B, self.B)
        self.assertEqual(state.K, K)
        self.assertTrue(state._has_perturbations)
        self.assertEqual(state.hand_states.shape, (self.B, K, self.T, self.D_hand))


class TestResultSelector(unittest.TestCase):
    """Tests for ResultSelector."""

    def setUp(self):
        self.B = 4
        self.K = 8
        self.T = 10
        self.D_hand = 25
        self.D_obj = 7

    def test_select_best_valid(self):
        """Test selecting best valid trajectory."""
        hand = torch.randn(self.B, self.K, self.T, self.D_hand)
        obj = torch.randn(self.B, self.K, self.T, self.D_obj)
        state = TrajectoryState(hand, obj)

        # Create energies - lower is better
        energies = torch.rand(self.B, self.K) * 10
        valid = torch.ones(self.B, self.K, dtype=torch.bool)

        # Make some invalid
        valid[0, :3] = False  # First 3 invalid for batch 0
        energies[0, 3] = 0.1  # Best valid for batch 0

        best_state, best_energy, success = ResultSelector.select_best_valid(state, energies, valid)

        self.assertEqual(best_state.B, self.B)
        self.assertEqual(best_state.T, self.T)
        self.assertFalse(best_state._has_perturbations)
        self.assertTrue(success.all())  # All batches have at least one valid

        # Check that best for batch 0 is from perturbation 3
        self.assertTrue(torch.allclose(best_state.hand_states[0], hand[0, 3]))

    def test_select_best_with_no_valid(self):
        """Test when a batch has no valid solutions."""
        hand = torch.randn(self.B, self.K, self.T, self.D_hand)
        obj = torch.randn(self.B, self.K, self.T, self.D_obj)
        state = TrajectoryState(hand, obj)

        energies = torch.rand(self.B, self.K)
        valid = torch.ones(self.B, self.K, dtype=torch.bool)
        valid[1, :] = False  # No valid for batch 1

        best_state, best_energy, success = ResultSelector.select_best_valid(state, energies, valid)

        self.assertTrue(success[0])
        self.assertFalse(success[1])  # Batch 1 failed
        self.assertTrue(success[2])
        self.assertTrue(success[3])


if __name__ == "__main__":
    unittest.main()
