"""
Trajectory optimization framework for video post-processing.

This module provides a modular, GPU-efficient optimization framework for
post-processing hand+object trajectories extracted from video.

Main components:
- TrajectoryState: Optimization variables (hand + object poses)
- ReferenceTrajectory: Original video trajectory (target)
- OptimizationContext: Shared context with models and cache
- CostFunction: Base class for costs
- Optimizer: Base class for optimizers
- OptimizationRunner: Main optimization loop

Example usage:
    from graspqp.optim import (
        TrajectoryState, ReferenceTrajectory, OptimizationContext,
        OptimizationProblem, OptimizationRunner
    )
    from graspqp.optim.costs import ReferenceTrackingCost, PenetrationCost
    from graspqp.optim.optimizers import AdamOptimizer
    from graspqp.optim.callbacks import LoggingCallback

    # Create reference from video
    reference = ReferenceTrajectory(
        hand_states=video_hand_poses,
        object_states=video_object_poses,
        contact_indices=contact_indices,
    )

    # Initialize state from reference
    state = TrajectoryState.from_reference(reference, n_perturbations=8)

    # Create context
    ctx = OptimizationContext(hand_model, object_model, reference)

    # Build problem
    problem = OptimizationProblem(ctx)
    problem.add_cost(ReferenceTrackingCost(weight=100.0))
    problem.add_cost(PenetrationCost(weight=1000.0))

    # Run optimization
    optimizer = AdamOptimizer(lr=0.01)
    runner = OptimizationRunner(problem, optimizer, [LoggingCallback()])
    optimized = runner.run(state, n_iters=2000)
"""

from .context import OptimizationContext
from .problem import OptimizationProblem
from .runner import OptimizationRunner
from .state import ReferenceTrajectory, ResultSelector, TrajectoryState

__all__ = [
    # State
    "TrajectoryState",
    "ReferenceTrajectory",
    "ResultSelector",
    # Context
    "OptimizationContext",
    # Problem
    "OptimizationProblem",
    # Runner
    "OptimizationRunner",
]
