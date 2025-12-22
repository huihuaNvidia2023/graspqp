"""
Logging callback.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Optional

from .base import Callback

if TYPE_CHECKING:
    from ..problem import OptimizationProblem
    from ..state import TrajectoryState


class LoggingCallback(Callback):
    """
    Callback for logging optimization progress.

    Args:
        log_every: Log every N steps (default: 100)
        print_fn: Function to use for printing (default: print)
        verbose: Whether to print detailed cost breakdown (default: True)
    """

    def __init__(
        self,
        log_every: int = 100,
        print_fn=None,
        verbose: bool = True,
    ):
        self.log_every = log_every
        self.print_fn = print_fn or print
        self.verbose = verbose

        self._start_time: Optional[float] = None
        self._step_times: list = []

    def on_optimization_start(self, state, problem, n_iters):
        self._start_time = time.time()
        self._step_times = []

        self.print_fn(f"Starting optimization: {n_iters} iterations")
        self.print_fn(f"  State shape: B={state.B}, T={state.T}, K={state.K}")
        self.print_fn(f"  Costs: {list(problem.costs.keys())}")

    def on_optimization_end(self, state, problem, n_iters):
        total_time = time.time() - self._start_time
        avg_step_time = sum(self._step_times) / len(self._step_times) if self._step_times else 0

        self.print_fn(f"\nOptimization complete:")
        self.print_fn(f"  Total time: {total_time:.2f}s")
        self.print_fn(f"  Avg step time: {avg_step_time*1000:.2f}ms")

        # Final cost breakdown
        breakdown = problem.cost_breakdown(state)
        self.print_fn(f"  Final costs:")
        for name, value in breakdown.items():
            self.print_fn(f"    {name}: {value:.4f}")

    def on_step_start(self, step, state, problem):
        self._step_start = time.time()

    def on_step_end(self, step, state, problem, energies):
        step_time = time.time() - self._step_start
        self._step_times.append(step_time)

        if step % self.log_every == 0 or step == 1:
            total_energy = sum(e.mean().item() for e in energies.values())
            elapsed = time.time() - self._start_time

            msg = f"Step {step}: total_energy={total_energy:.4f}, time={elapsed:.1f}s"

            if self.verbose:
                breakdown = ", ".join(f"{k}={v.mean().item():.4f}" for k, v in energies.items())
                msg += f" | {breakdown}"

            self.print_fn(msg)
