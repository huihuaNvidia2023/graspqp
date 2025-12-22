#!/usr/bin/env python
"""
Example script for trajectory optimization.

This script demonstrates how to use the new optimization framework
for post-processing hand+object trajectories extracted from video.

Usage:
    python scripts/trajectory_optimize.py \
        --config configs/trajectory_optimization.yaml \
        --input trajectory_data.pt \
        --output optimized.pt \
        --n_perturbations 8
"""

import argparse
from pathlib import Path

import torch

from graspqp.optim import (OptimizationContext, OptimizationProblem, OptimizationRunner, ReferenceTrajectory,
                           ResultSelector, TrajectoryState)
from graspqp.optim.callbacks import CheckpointCallback, LoggingCallback
from graspqp.optim.costs import AccelerationCost, PenetrationCost, ReferenceTrackingCost, VelocitySmoothnessCost
from graspqp.optim.optimizers import AdamOptimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Trajectory Optimization")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--input", type=str, required=True, help="Input trajectory file")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--n_perturbations", type=int, default=8, help="Number of perturbations per trajectory")
    parser.add_argument("--n_iters", type=int, default=2000, help="Number of optimization iterations")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()


def create_dummy_data(B: int = 4, T: int = 20, D_hand: int = 25, D_obj: int = 7, n_contacts: int = 8):
    """Create dummy trajectory data for testing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simulate a smooth trajectory with some noise
    t = torch.linspace(0, 1, T).to(device)

    # Hand states: position + rotation + joints
    # Create a simple trajectory and add noise
    hand_base = torch.zeros(B, T, D_hand, device=device)
    hand_base[:, :, 0] = t.unsqueeze(0) * 0.1  # Move in x direction
    hand_noise = torch.randn(B, T, D_hand, device=device) * 0.02
    hand_states = hand_base + hand_noise

    # Object states: position + quaternion
    obj_base = torch.zeros(B, T, D_obj, device=device)
    obj_base[:, :, 3] = 1.0  # w component of quaternion
    obj_noise = torch.randn(B, T, D_obj, device=device) * 0.01
    object_states = obj_base + obj_noise

    # Contact indices (fixed for trajectory)
    contact_indices = torch.randint(0, 100, (B, n_contacts), device=device)

    return hand_states, object_states, contact_indices


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load or create trajectory data
    input_path = Path(args.input)
    if input_path.exists():
        print(f"Loading trajectory data from {input_path}")
        data = torch.load(input_path, map_location=device)
        hand_states = data["hand_states"]
        object_states = data["object_states"]
        contact_indices = data["contact_indices"]
    else:
        print("Input file not found, creating dummy data for demonstration...")
        hand_states, object_states, contact_indices = create_dummy_data()

    B, T, D_hand = hand_states.shape
    D_obj = object_states.shape[-1]
    print(f"Trajectory shape: B={B}, T={T}, D_hand={D_hand}, D_obj={D_obj}")

    # Create reference trajectory (the video data)
    reference = ReferenceTrajectory(
        hand_states=hand_states,
        object_states=object_states,
        contact_indices=contact_indices,
        dt=0.1,
    )

    # Create initial state with perturbations
    K = args.n_perturbations
    initial_state = TrajectoryState.from_reference(
        reference,
        n_perturbations=K,
        perturbation_scale=0.01,
    )
    print(f"Created {K} perturbations per trajectory: state shape = (B={B}, K={K}, T={T})")

    # Create context (would normally include hand and object models)
    # For this example, we use mock models
    class MockModel:
        def set_parameters(self, params, **kwargs):
            pass

        def get_surface_points(self):
            n = params.shape[0]
            return torch.zeros(n, 100, 3, device=params.device)

        def cal_distance(self, points):
            return torch.ones(points.shape[0], points.shape[1], device=points.device), None

    ctx = OptimizationContext(
        hand_model=MockModel(),
        object_model=MockModel(),
        reference=reference,
        device=device,
    )

    # Build optimization problem
    problem = OptimizationProblem(ctx)
    problem.add_cost(ReferenceTrackingCost(weight=100.0))
    problem.add_cost(VelocitySmoothnessCost(weight=1.0))
    problem.add_cost(AccelerationCost(weight=0.1))
    # Note: PenetrationCost would require real hand/object models

    print(f"\nOptimization problem:")
    print(problem)

    # Create optimizer and runner
    optimizer = AdamOptimizer(lr=args.lr)
    callbacks = [
        LoggingCallback(log_every=100),
    ]
    runner = OptimizationRunner(problem, optimizer, callbacks=callbacks)

    # Run optimization
    print(f"\nStarting optimization for {args.n_iters} iterations...")
    optimized_state = runner.run(initial_state, n_iters=args.n_iters)

    # Select best valid result per trajectory
    with torch.no_grad():
        energies = problem.total_energy(optimized_state)
        # Reshape to (B, K)
        energies_bk = energies.reshape(B, K)
        # All are "valid" for this demo
        valid = torch.ones(B, K, dtype=torch.bool, device=device)

        best_state, best_energy, success = ResultSelector.select_best_valid(optimized_state, energies_bk, valid)

    print(f"\n=== Results ===")
    print(f"Successful trajectories: {success.sum().item()}/{B}")
    print(f"Best energies: {best_energy.tolist()}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "optimized_trajectory": best_state.to_trajectory_dict(),
        "reference_trajectory": {
            "hand_states": reference.hand_states.cpu(),
            "object_states": reference.object_states.cpu(),
        },
        "energies": best_energy.cpu(),
        "success": success.cpu(),
        "config": {
            "n_perturbations": K,
            "n_iters": args.n_iters,
            "lr": args.lr,
        },
    }

    torch.save(result, output_path)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
