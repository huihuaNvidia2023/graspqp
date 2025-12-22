"""
Unit tests for OptimizationRunner.
"""

import unittest
from unittest.mock import MagicMock

import torch

from graspqp.optim.callbacks import Callback, LoggingCallback
from graspqp.optim.context import OptimizationContext
from graspqp.optim.costs import VelocitySmoothnessCost
from graspqp.optim.optimizers import AdamOptimizer
from graspqp.optim.problem import OptimizationProblem
from graspqp.optim.runner import OptimizationRunner
from graspqp.optim.state import TrajectoryState


class CountingCallback(Callback):
    """Callback that counts calls for testing."""

    def __init__(self):
        self.start_count = 0
        self.end_count = 0
        self.step_start_count = 0
        self.step_end_count = 0

    def on_optimization_start(self, state, problem, n_iters):
        self.start_count += 1

    def on_optimization_end(self, state, problem, n_iters):
        self.end_count += 1

    def on_step_start(self, step, state, problem):
        self.step_start_count += 1

    def on_step_end(self, step, state, problem, energies):
        self.step_end_count += 1


class EarlyStopCallback(Callback):
    """Callback that stops after N steps."""

    def __init__(self, stop_after: int):
        self.stop_after = stop_after

    def should_stop(self, step, state, energies):
        if step >= self.stop_after:
            return True, "Test early stop"
        return False, ""


class TestOptimizationRunner(unittest.TestCase):
    """Tests for OptimizationRunner."""

    def setUp(self):
        self.B = 4
        self.T = 10
        self.D_hand = 25
        self.D_obj = 7

        self.ctx = OptimizationContext(
            hand_model=MagicMock(),
            object_model=MagicMock(),
        )

        self.problem = OptimizationProblem(self.ctx)
        self.problem.add_cost(VelocitySmoothnessCost(name="vel", weight=1.0))

        self.optimizer = AdamOptimizer(lr=0.01)

    def test_basic_run(self):
        """Test basic optimization run."""
        runner = OptimizationRunner(self.problem, self.optimizer)

        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)
        state = TrajectoryState(hand, obj)

        result = runner.run(state, n_iters=10, show_progress=False)

        self.assertIsInstance(result, TrajectoryState)
        self.assertEqual(result.B, self.B)
        self.assertEqual(result.T, self.T)

    def test_callbacks_called(self):
        """Test that callbacks are called correctly."""
        callback = CountingCallback()
        runner = OptimizationRunner(self.problem, self.optimizer, callbacks=[callback])

        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)
        state = TrajectoryState(hand, obj)

        runner.run(state, n_iters=5, show_progress=False)

        self.assertEqual(callback.start_count, 1)
        self.assertEqual(callback.end_count, 1)
        self.assertEqual(callback.step_start_count, 5)
        self.assertEqual(callback.step_end_count, 5)

    def test_early_stopping(self):
        """Test early stopping via callback."""
        early_stop = EarlyStopCallback(stop_after=3)
        counting = CountingCallback()

        runner = OptimizationRunner(self.problem, self.optimizer, callbacks=[early_stop, counting])

        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)
        state = TrajectoryState(hand, obj)

        runner.run(state, n_iters=10, show_progress=False)

        # Should have stopped after 3 steps
        self.assertEqual(counting.step_end_count, 3)

    def test_single_step(self):
        """Test single step execution."""
        runner = OptimizationRunner(self.problem, self.optimizer)

        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)
        state = TrajectoryState(hand, obj)

        new_state = runner.step(state)

        self.assertIsInstance(new_state, TrajectoryState)

    def test_reset(self):
        """Test runner reset."""
        runner = OptimizationRunner(self.problem, self.optimizer)

        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)
        state = TrajectoryState(hand, obj)

        runner.run(state, n_iters=5, show_progress=False)

        self.assertEqual(self.optimizer.step_count, 5)

        runner.reset()

        self.assertEqual(self.optimizer.step_count, 0)


class TestOptimizationRunnerWithLogging(unittest.TestCase):
    """Test runner with logging callback."""

    def setUp(self):
        self.B = 4
        self.T = 10
        self.D_hand = 25
        self.D_obj = 7

        self.ctx = OptimizationContext(
            hand_model=MagicMock(),
            object_model=MagicMock(),
        )

        self.problem = OptimizationProblem(self.ctx)
        self.problem.add_cost(VelocitySmoothnessCost(name="vel", weight=1.0))

    def test_with_logging_callback(self):
        """Test that logging callback works."""
        logged_messages = []

        def mock_print(msg):
            logged_messages.append(msg)

        logging_cb = LoggingCallback(log_every=2, print_fn=mock_print)
        optimizer = AdamOptimizer(lr=0.01)
        runner = OptimizationRunner(self.problem, optimizer, callbacks=[logging_cb])

        hand = torch.randn(self.B, self.T, self.D_hand)
        obj = torch.randn(self.B, self.T, self.D_obj)
        state = TrajectoryState(hand, obj)

        runner.run(state, n_iters=5, show_progress=False)

        # Should have logged start, steps 1, 2, 4, and end
        self.assertTrue(any("Starting optimization" in msg for msg in logged_messages))
        self.assertTrue(any("Optimization complete" in msg for msg in logged_messages))


if __name__ == "__main__":
    unittest.main()
