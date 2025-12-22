"""
Unit tests for callbacks.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock

import torch

from graspqp.optim.callbacks import Callback, CallbackList, CheckpointCallback, LoggingCallback
from graspqp.optim.callbacks.checkpoint import EarlyStoppingCallback
from graspqp.optim.state import TrajectoryState


class TestCallback(unittest.TestCase):
    """Tests for base Callback class."""

    def test_default_methods_do_nothing(self):
        """Test that default callback methods do nothing."""
        cb = Callback()

        # These should not raise
        cb.on_optimization_start(None, None, 100)
        cb.on_optimization_end(None, None, 100)
        cb.on_step_start(1, None, None)
        cb.on_step_end(1, None, None, {})

        should_stop, reason = cb.should_stop(1, None, {})
        self.assertFalse(should_stop)
        self.assertEqual(reason, "")


class TestCallbackList(unittest.TestCase):
    """Tests for CallbackList."""

    def test_empty_list(self):
        """Test empty callback list."""
        cb_list = CallbackList()

        # Should not raise
        cb_list.on_optimization_start(None, None, 100)

    def test_append(self):
        """Test appending callbacks."""
        cb_list = CallbackList()
        cb = Callback()

        cb_list.append(cb)

        self.assertEqual(len(cb_list.callbacks), 1)

    def test_calls_all_callbacks(self):
        """Test that all callbacks are called."""
        call_counts = [0, 0]

        class CountingCallback1(Callback):
            def on_step_end(self, step, state, problem, energies):
                call_counts[0] += 1

        class CountingCallback2(Callback):
            def on_step_end(self, step, state, problem, energies):
                call_counts[1] += 1

        cb_list = CallbackList([CountingCallback1(), CountingCallback2()])
        cb_list.on_step_end(1, None, None, {})

        self.assertEqual(call_counts[0], 1)
        self.assertEqual(call_counts[1], 1)

    def test_should_stop_any(self):
        """Test that should_stop returns True if any callback returns True."""

        class StopCallback(Callback):
            def should_stop(self, step, state, energies):
                return True, "Stop now"

        cb_list = CallbackList([Callback(), StopCallback()])

        should_stop, reason = cb_list.should_stop(1, None, {})

        self.assertTrue(should_stop)
        self.assertEqual(reason, "Stop now")


class TestLoggingCallback(unittest.TestCase):
    """Tests for LoggingCallback."""

    def test_logs_at_correct_intervals(self):
        """Test that logging happens at correct intervals."""
        messages = []
        cb = LoggingCallback(log_every=3, print_fn=lambda x: messages.append(x))

        problem = MagicMock()
        problem.cost_breakdown = MagicMock(return_value={"test": 1.0})

        B, T, D = 4, 10, 25
        hand = torch.randn(B, T, D)
        obj = torch.randn(B, T, 7)
        state = TrajectoryState(hand, obj)

        energies = {"test": torch.ones(B)}

        cb.on_optimization_start(state, problem, 10)

        for step in [1, 2, 3, 4, 5, 6]:
            cb.on_step_start(step, state, problem)
            cb.on_step_end(step, state, problem, energies)

        # Should log at step 1, 3, 6
        step_logs = [m for m in messages if m.startswith("Step")]
        self.assertEqual(len(step_logs), 3)


class TestCheckpointCallback(unittest.TestCase):
    """Tests for CheckpointCallback."""

    def test_saves_checkpoints(self):
        """Test that checkpoints are saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = CheckpointCallback(save_every=5, save_dir=tmpdir, save_best=True)

            B, T, D = 4, 10, 25
            hand = torch.randn(B, T, D)
            obj = torch.randn(B, T, 7)
            state = TrajectoryState(hand, obj)

            problem = MagicMock()
            energies = {"test": torch.ones(B) * 10}

            cb.on_optimization_start(state, problem, 10)

            for step in [1, 2, 3, 4, 5]:
                cb.on_step_end(step, state, problem, energies)

            cb.on_optimization_end(state, problem, 5)

            # Check files exist
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "checkpoint_step_5.pt")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "checkpoint_final.pt")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "checkpoint_best.pt")))


class TestEarlyStoppingCallback(unittest.TestCase):
    """Tests for EarlyStoppingCallback."""

    def test_stops_on_plateau(self):
        """Test that optimization stops on energy plateau."""
        cb = EarlyStoppingCallback(patience=3, min_delta=0.1)

        B = 4
        state = MagicMock()

        # Initialize
        cb.on_optimization_start(state, None, 100)

        # Energy decreases - should not stop
        energies = {"total": torch.ones(B) * 10}
        self.assertFalse(cb.should_stop(1, state, energies)[0])

        energies = {"total": torch.ones(B) * 9}
        self.assertFalse(cb.should_stop(2, state, energies)[0])

        energies = {"total": torch.ones(B) * 8}
        self.assertFalse(cb.should_stop(3, state, energies)[0])

        # Energy plateaus - should stop after patience
        energies = {"total": torch.ones(B) * 8.05}  # Less than min_delta improvement
        self.assertFalse(cb.should_stop(4, state, energies)[0])

        energies = {"total": torch.ones(B) * 8.03}
        self.assertFalse(cb.should_stop(5, state, energies)[0])

        energies = {"total": torch.ones(B) * 8.04}
        should_stop, reason = cb.should_stop(6, state, energies)

        self.assertTrue(should_stop)
        self.assertIn("no improvement", reason)

    def test_resets_patience_on_improvement(self):
        """Test that patience resets when energy improves."""
        cb = EarlyStoppingCallback(patience=3, min_delta=0.1)

        B = 4
        state = MagicMock()

        cb.on_optimization_start(state, None, 100)

        # Energy decreases
        energies = {"total": torch.ones(B) * 10}
        cb.should_stop(1, state, energies)

        # Plateau for 2 steps
        energies = {"total": torch.ones(B) * 10}
        cb.should_stop(2, state, energies)
        cb.should_stop(3, state, energies)

        # Improve! Should reset patience
        energies = {"total": torch.ones(B) * 9}
        cb.should_stop(4, state, energies)

        # Plateau again - should take another 3 steps to stop
        energies = {"total": torch.ones(B) * 9}
        self.assertFalse(cb.should_stop(5, state, energies)[0])
        self.assertFalse(cb.should_stop(6, state, energies)[0])
        self.assertTrue(cb.should_stop(7, state, energies)[0])


if __name__ == "__main__":
    unittest.main()
