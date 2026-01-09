#!/usr/bin/env python
"""
Sanity check for AdamOptimizer - debug why gradient descent isn't working.

Based on sanity_check_optim.py but uses AdamOptimizer instead of MalaStarOptimizer.
This script helps debug gradient flow issues.

Usage:
    python scripts/sanity_check_adam.py \
        --object_code_list apple \
        --batch_size 4 \
        --n_iter 200 \
        --prior_file configs/extracted_prior.yaml \
        --init_mode prior \
        --lr 0.01
"""

import argparse
import math
import os

import numpy as np
import roma
import torch
from tqdm import tqdm

from graspqp.core import GraspPriorLoader, ObjectModel
from graspqp.core.initializations import initialize_convex_hull
from graspqp.hands import AVAILABLE_HANDS, get_hand_model
from graspqp.metrics import GraspSpanMetricFactory
from graspqp.optim.context import OptimizationContext
from graspqp.optim.costs.grasp import ContactDistanceCost, ForceClosureCost, JointLimitCost, PriorPoseCost
from graspqp.optim.costs.penetration import PenetrationCost, SelfPenetrationCost
from graspqp.optim.optimizers.torch_optim import AdamOptimizer
from graspqp.optim.problem import OptimizationProblem
from graspqp.optim.state import ReferenceTrajectory, TrajectoryState
from graspqp.utils.profiler import get_profiler
from graspqp.utils.transforms import robust_compute_rotation_matrix_from_ortho6d


def parse_args():
    parser = argparse.ArgumentParser(description="Adam Optimizer Sanity Check")
    parser.add_argument("--seed", default=1, type=int, help="Random seed for reproducibility")
    parser.add_argument("--object_code_list", default=["apple"], nargs="+", help="List of object codes to optimize grasps for")
    parser.add_argument("--n_contact", default=8, type=int, help="Number of contact points per grasp")
    parser.add_argument("--batch_size", default=4, type=int, help="Number of grasp samples per object")
    parser.add_argument("--n_iter", default=200, type=int, help="Number of optimization iterations")

    # Weights
    parser.add_argument("--w_dis", default=10.0, type=float, help="Weight for contact distance cost")
    parser.add_argument("--w_fc", default=1.0, type=float, help="Weight for force closure cost (0 to disable)")
    parser.add_argument("--w_pen", default=10.0, type=float, help="Weight for hand-object penetration cost")
    parser.add_argument("--w_spen", default=10.0, type=float, help="Weight for self-penetration cost")
    parser.add_argument("--w_joints", default=1.0, type=float, help="Weight for joint limit violation cost")
    parser.add_argument("--w_prior", default=10.0, type=float, help="Weight for prior pose deviation cost")
    parser.add_argument("--w_svd", default=0.1, type=float, help="SVD regularization gain for force closure QP")

    # Adam settings
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate for Adam optimizer")
    parser.add_argument("--beta1", default=0.9, type=float, help="Adam beta1 (exponential decay for 1st moment)")
    parser.add_argument("--beta2", default=0.999, type=float, help="Adam beta2 (exponential decay for 2nd moment)")

    # Adaptive contact resampling
    parser.add_argument("--resample_contacts", action="store_true", help="Enable contact resampling for stuck batches")
    parser.add_argument("--resample_interval", default=50, type=int, help="Check for stuck batches every N steps")
    parser.add_argument("--resample_threshold", default=3.0, type=float, help="Resample if energy > best * threshold")

    # Optimization variables
    parser.add_argument(
        "--optimize_object",
        action="store_true",
        help="Optimize object_states in addition to hand_states. "
        "Default is False (only optimize hand-to-object transform, object fixed at origin)",
    )

    # Initialization
    parser.add_argument("--jitter_strength", default=0.1, type=float)
    parser.add_argument("--distance_lower", default=0.05, type=float)
    parser.add_argument("--distance_upper", default=0.1, type=float)
    parser.add_argument("--rotate_lower", default=-180 * math.pi / 180, type=float)
    parser.add_argument("--rotate_upper", default=180 * math.pi / 180, type=float)
    parser.add_argument("--pitch_lower", default=-15 * math.pi / 180, type=float)
    parser.add_argument("--pitch_upper", default=15 * math.pi / 180, type=float)
    parser.add_argument("--tilt_lower", default=-45 * math.pi / 180, type=float)
    parser.add_argument("--tilt_upper", default=45 * math.pi / 180, type=float)

    parser.add_argument("--data_root_path", default="./objects", type=str)
    parser.add_argument("--hand_name", default="allegro", type=str, choices=AVAILABLE_HANDS)
    parser.add_argument("--prior_file", default=None, type=str)
    parser.add_argument("--mesh_extension", default=".obj", type=str)
    parser.add_argument("--grasp_type", default="all", type=str)
    parser.add_argument("--friction", default=0.2, type=float)
    parser.add_argument("--max_lambda_limit", default=20.0, type=float)
    parser.add_argument("--n_friction_cone", default=4, type=int)

    parser.add_argument(
        "--init_mode",
        default="prior",
        type=str,
        choices=["convex_hull", "prior"],
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug output")
    parser.add_argument("--profile", action="store_true", help="Enable detailed profiling")
    parser.add_argument(
        "--fc_threshold",
        default=0.01,
        type=float,
        help="Contact distance threshold for force closure (None=always compute)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seeds
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize profiler
    profiler = get_profiler(enabled=args.profile, cuda_sync=True)

    num_objects = len(args.object_code_list)
    total_batch_size = num_objects * args.batch_size

    print("=" * 70)
    print("Adam Optimizer Sanity Check")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Objects: {args.object_code_list}")
    print(f"Batch size: {total_batch_size}")
    print(f"Iterations: {args.n_iter}")
    print(f"Learning rate: {args.lr}")
    if args.profile:
        print("Profiling: ENABLED")
    print(f"FC threshold: {args.fc_threshold}")
    print("=" * 70)

    # =========================================================================
    # 1. Initialize hand and object models
    # =========================================================================
    hand_model = get_hand_model(args.hand_name, device, grasp_type=args.grasp_type)

    object_model = ObjectModel(
        data_root_path=args.data_root_path,
        batch_size_each=args.batch_size,
        num_samples=1000,
        device=device,
    )
    object_model.initialize(args.object_code_list, extension=args.mesh_extension)

    # Initialize with convex_hull (for random state)
    initialize_convex_hull(hand_model, object_model, args)

    # =========================================================================
    # 2. Load prior pose and object state
    # =========================================================================
    prior_pose = None
    prior_object_state = None
    if args.prior_file is not None:
        print(f"\nLoading prior from: {args.prior_file}")
        prior_config = GraspPriorLoader.load_from_file(args.prior_file)

        if prior_config.hands:  # Use .hands (new name) or .priors (legacy alias)
            prior_data = GraspPriorLoader.expand_priors(prior_config, total_batch_size, hand_model, device)
            prior_pose = GraspPriorLoader.create_hand_pose_from_priors(prior_data)
            prior_object_state = GraspPriorLoader.create_object_state_from_priors(prior_data)
            print(f"  Loaded hand prior: shape={prior_pose.shape}")
            print(f"  Loaded object state: shape={prior_object_state.shape}")
            print(f"    Object translation: {prior_object_state[0, :3].tolist()}")
            print(f"    Object rotation (qxyzw): {prior_object_state[0, 3:7].tolist()}")

    # Override with prior if using prior init mode
    if args.init_mode == "prior" and prior_pose is not None:
        print(f"\nInit mode: prior - setting hand pose from prior")
        hand_model.set_parameters(prior_pose, hand_model.contact_point_indices)

    print(f"\nn_contact_candidates: {hand_model.n_contact_candidates}")

    # =========================================================================
    # 3. Create TrajectoryState (T=1 single frame)
    # =========================================================================
    D_hand = hand_model.hand_pose.shape[1]
    D_obj = 7

    initial_hand = hand_model.hand_pose.detach().clone()

    # Get object state from prior if available, otherwise use default (origin)
    if prior_object_state is not None:
        object_state = prior_object_state.unsqueeze(1)  # (B, 1, 7)
    else:
        object_state = GraspPriorLoader.get_default_object_state(total_batch_size, device).unsqueeze(1)

    state = TrajectoryState(
        hand_states=initial_hand.unsqueeze(1),
        object_states=object_state,
    )
    print(f"\nTrajectoryState: B={state.B}, T={state.T}, D_hand={state.D_hand}")

    # =========================================================================
    # 4. Create OptimizationContext
    # =========================================================================
    # Get contact finger constraints from prior config
    contact_fingers = None
    contact_mode = "uniform"  # Default
    if args.prior_file is not None and prior_config is not None:
        contact_mode = prior_config.contact.mode
        # Check global contact config first
        if prior_config.contact.links:
            contact_fingers = prior_config.contact.links
        # Per-hand contact_links override (use first hand's config)
        elif prior_config.hands and prior_config.hands[0].contact_links:
            contact_fingers = prior_config.hands[0].contact_links

    if contact_fingers:
        print(f"  Contact mode: {contact_mode}")
        print(f"  Contact fingers from prior: {contact_fingers}")
    else:
        print(f"  Contact mode: {contact_mode} (all fingers)")

    if prior_pose is not None:
        reference = ReferenceTrajectory(
            hand_states=prior_pose.unsqueeze(1),
            object_states=object_state.clone(),
            contact_fingers=contact_fingers,
            n_contacts=args.n_contact,
            hand_type=args.hand_name,
        )
    else:
        reference = ReferenceTrajectory(
            hand_states=initial_hand.unsqueeze(1),
            object_states=object_state.clone(),
            contact_fingers=contact_fingers,
            n_contacts=args.n_contact,
            hand_type=args.hand_name,
        )

    context = OptimizationContext(
        hand_model=hand_model,
        object_model=object_model,
        reference=reference,
        device=device,
        profiler=profiler,
    )

    # Set up contact sampler based on prior config
    # Only create constrained sampler if contact_fingers are specified
    if contact_fingers:
        sampler = context.create_contact_sampler_from_reference()
        if context.contact_sampler is not None:
            print(f"  Contact sampler: constrained to specified fingers")
        else:
            print(f"  Contact sampler: failed to create, falling back to uniform")
    else:
        # Use uniform sampling (no finger constraints) - don't infer from random contacts
        sampler = None
        print(f"  Contact sampler: uniform (all contact candidates)")

    # =========================================================================
    # 5. Create OptimizationProblem with costs
    # =========================================================================
    problem = OptimizationProblem(context, profiler=profiler)

    problem.add_cost(ContactDistanceCost(name="contact_distance", weight=args.w_dis))

    if args.w_fc > 0:
        # fc_threshold=None means always compute QP (for comparison)
        # fc_threshold=0.01 means only compute when contacts established
        fc_config = {
            "svd_gain": args.w_svd,
            "contact_threshold": args.fc_threshold if args.fc_threshold > 0 else None,
        }
        fc_cost = ForceClosureCost(name="force_closure", weight=args.w_fc, config=fc_config)
        energy_fnc = GraspSpanMetricFactory.create(
            GraspSpanMetricFactory.MetricType.GRASPQP,
            solver_kwargs={
                "friction": args.friction,
                "max_limit": args.max_lambda_limit,
                "n_cone_vecs": args.n_friction_cone,
            },
        )
        fc_cost.set_energy_function(energy_fnc)
        problem.add_cost(fc_cost)

    problem.add_cost(PenetrationCost(name="penetration", weight=args.w_pen))
    problem.add_cost(SelfPenetrationCost(name="self_penetration", weight=args.w_spen))
    problem.add_cost(JointLimitCost(name="joint_limits", weight=args.w_joints))

    if prior_pose is not None and args.w_prior > 0:
        prior_cost = PriorPoseCost(name="prior_pose", weight=args.w_prior)
        prior_cost.set_prior_pose(prior_pose)
        problem.add_cost(prior_cost)

    print("\nCosts configured:")
    for name, cost in problem.costs.items():
        print(f"  {name}: weight={cost.weight}")

    # =========================================================================
    # 6. Create AdamOptimizer
    # =========================================================================
    optimizer = AdamOptimizer(
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        debug=args.debug,
        optimize_object=args.optimize_object,
        resample_contacts=args.resample_contacts,
        resample_interval=args.resample_interval,
        resample_threshold=args.resample_threshold,
    )
    opt_vars = "hand + object" if args.optimize_object else "hand only (object fixed at origin)"
    print(f"\nOptimizer: AdamOptimizer (lr={args.lr}, optimizing: {opt_vars})")
    if args.resample_contacts:
        print(
            f"  Contact resampling: enabled (interval={args.resample_interval}, threshold={args.resample_threshold}x)"
        )

    # IMPORTANT: Initialize optimizer with state (creates persistent params)
    print("\nInitializing optimizer...")
    state = optimizer.initialize(state)
    print(f"  Optimizer initialized with persistent parameters")

    # =========================================================================
    # 7. Compute initial energy
    # =========================================================================
    with torch.no_grad():
        initial_costs = problem.evaluate_all(state)
        initial_energy = problem.total_energy(state)

    print(f"\nInitial energy: mean={initial_energy.mean().item():.2f}, best={initial_energy.min().item():.2f}")
    print("Initial cost breakdown:")
    for k, v in initial_costs.items():
        print(f"  {k}: mean={v.mean().item():.4f}")

    # =========================================================================
    # 8. Main optimization loop with detailed debugging
    # =========================================================================
    print(f"\nStarting optimization...")

    energy_history = []

    import time

    start_time = time.perf_counter()

    for step in tqdm(range(1, args.n_iter + 1), desc="Optimizing"):
        with profiler.section("step"):
            # Clear step cache
            context.clear_step_cache()

            # Store old state for comparison
            old_hand = state.hand_states.detach().clone()

            # Optimizer step
            with profiler.section("optimizer_step"):
                state = optimizer.step(state, problem)

            # Compute energy for logging
            with torch.no_grad():
                energy = problem.total_energy(state)
                energy_history.append(energy.mean().item())

            # Check hand state change
            hand_change = (state.hand_states - old_hand).abs().mean().item()

            # Periodic detailed logging
            if step % 50 == 0 or step <= 5:
                print(f"\nStep {step}:")
                print(f"  Energy: mean={energy.mean().item():.4f}, best={energy.min().item():.4f}")
                print(f"  Hand state change: {hand_change:.6f}")

                # Note: After optimizer.step(), the internal params are new tensors
                # (created via detach().clone()), so grad is None. This is expected.
                # The gradient existed during backward() but is cleared when new tensors are created.
                # Energy decrease confirms gradient flow is working.

                # Per-cost breakdown
                with torch.no_grad():
                    costs = problem.evaluate_all(state)
                    for k, v in costs.items():
                        print(f"    {k}: {v.mean().item():.4f}")

        profiler.step_done()

    total_time = time.perf_counter() - start_time

    # =========================================================================
    # 9. Final results
    # =========================================================================
    with torch.no_grad():
        final_costs = problem.evaluate_all(state)
        final_energy = problem.total_energy(state)

    # =========================================================================
    # 10. Export results
    # =========================================================================
    # Update hand_model with final state for export
    # IMPORTANT: detach() to remove gradient tracking before saving
    final_hand = state.hand_states.squeeze(1).detach()  # (B, D)
    hand_model.set_parameters(final_hand, hand_model.contact_point_indices)

    # Create output directory
    output_dir = os.path.join(
        args.data_root_path,
        args.object_code_list[0],
        "grasp_predictions",
        args.hand_name,
        f"{args.n_contact}_contacts",
        "adam_sanity_check",
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save checkpoint
    hand_poses = robust_compute_rotation_matrix_from_ortho6d(final_hand[:, 3:9])
    hand_qxyzw = roma.rotmat_to_unitquat(hand_poses)
    hand_qwxyz = hand_qxyzw[:, [3, 0, 1, 2]]
    root_pose = torch.cat([final_hand[:, :3], hand_qwxyz], dim=1)

    # Find object mesh path for viewer
    object_mesh_path = os.path.join(args.data_root_path, args.object_code_list[0], f"{args.object_code_list[0]}_scaled.obj")
    if not os.path.exists(object_mesh_path):
        # Try without _scaled suffix
        object_mesh_path = os.path.join(args.data_root_path, args.object_code_list[0], f"{args.object_code_list[0]}.obj")

    data = {
        "values": final_energy.detach().cpu(),
        "parameters": {
            "root_pose": root_pose.detach().cpu(),
            **{name: final_hand[:, 9 + i].detach().cpu() for i, name in enumerate(hand_model._actuated_joints_names)},
        },
        "contact_idx": hand_model.contact_point_indices.detach().cpu(),
        "contact_links": hand_model._contact_links,
        "metadata": {
            "hand_name": args.hand_name,
            "object_code": args.object_code_list[0],
            "data_root_path": os.path.abspath(args.data_root_path),
            "object_mesh_path": os.path.abspath(object_mesh_path) if os.path.exists(object_mesh_path) else "",
        },
    }

    file_path = os.path.join(output_dir, f"{args.object_code_list[0]}.dexgrasp.pt")
    torch.save(data, file_path)
    print(f"\n\033[94m==> Exported to {os.path.abspath(file_path)}\033[0m")

    print("\n" + "=" * 70)
    print("=== Final Results ===")
    print("=" * 70)
    print(f"Energy: best={final_energy.min().item():.2f}, mean={final_energy.mean().item():.2f}")
    print("Breakdown:")
    for k, v in final_costs.items():
        print(f"  {k}: mean={v.mean().item():.4f}")

    # Summary of optimization
    print("\n=== Optimization Summary ===")
    print(f"Initial energy: {energy_history[0]:.4f}")
    print(f"Final energy: {energy_history[-1]:.4f}")
    print(
        f"Energy reduction: {energy_history[0] - energy_history[-1]:.4f} ({(1 - energy_history[-1]/energy_history[0])*100:.1f}%)"
    )
    print(f"\nTiming: {total_time:.2f}s total, {total_time * 1000 / args.n_iter:.2f}ms/iter")

    # Profiler summary
    if args.profile:
        print("\n=== Profiler Summary ===")
        profiler.summary()

    # Check if stuck
    if len(energy_history) > 50:
        recent_change = abs(energy_history[-1] - energy_history[-50])
        if recent_change < 0.01:
            print("\n⚠️  WARNING: Optimization appears STUCK (energy unchanged in last 50 steps)")
            print("    Possible causes:")
            print("    1. Gradient flow is broken (set_parameters cloning tensors)")
            print("    2. Learning rate too small")
            print("    3. Cost functions not differentiable")
            print("    4. Local minimum reached")


if __name__ == "__main__":
    main()
