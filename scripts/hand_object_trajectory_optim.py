#!/usr/bin/env python
"""
Hand-Object Trajectory Optimization for T=2 frames.

This script optimizes a two-frame grasp trajectory where:
- Frame 0: Approach pose (hand approaching object)
- Frame 1: Contact pose (hand in grasp position)

The optimization enforces:
- Physical constraints at each frame (penetration, contact, force closure)
- Temporal smoothness between frames
- Prior tracking to stay close to reference trajectory

Output format follows docs/OPTIMIZATION_FRAMEWORK_DESIGN.md:
- Legacy fields for backward compatibility with visualize_result.py
- Trajectory data with all T frames
- Cost breakdown and diagnostics

Usage:
    python scripts/hand_object_trajectory_optim.py \
        --data_root_path ./objects \
        --object_code_list apple \
        --hand_name allegro \
        --batch_size 4 \
        --n_iter 200 \
        --prior_file configs/two_frame_prior.yaml
"""

import argparse
import json
import math
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import roma
import torch
from tqdm import tqdm

from graspqp.core import GraspPriorLoader, ObjectModel
from graspqp.hands import AVAILABLE_HANDS, get_hand_model
from graspqp.metrics import GraspSpanMetricFactory
from graspqp.optim.context import OptimizationContext
from graspqp.optim.costs.grasp import ContactDistanceCost, ForceClosureCost, JointLimitCost, PriorPoseCost
from graspqp.optim.costs.penetration import PenetrationCost, SelfPenetrationCost
from graspqp.optim.costs.reference import ReferenceTrackingCost
from graspqp.optim.costs.temporal import VelocitySmoothnessCost
from graspqp.optim.optimizers.mala_star import MalaStarOptimizer
from graspqp.optim.optimizers.mala_star_trajectory import MalaStarTrajectoryOptimizer
from graspqp.optim.optimizers.torch_optim import AdamOptimizer
from graspqp.optim.problem import OptimizationProblem
from graspqp.optim.state import ReferenceTrajectory, TrajectoryState
from graspqp.utils.profiler import get_profiler
from graspqp.utils.transforms import robust_compute_rotation_matrix_from_ortho6d


def parse_args():
    parser = argparse.ArgumentParser(description="Hand-Object Trajectory Optimization (T=2)")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--object_code_list", default=["apple"], nargs="+")
    parser.add_argument("--n_contact", default=8, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--n_iter", default=200, type=int)
    parser.add_argument("--n_frames", default=2, type=int, help="Number of frames (T)")

    # Weights - DEFAULT to sanity_check_optim.py values for baseline comparison
    # These match fit.py / sanity_check_optim.py defaults
    parser.add_argument("--w_dis", default=100.0, type=float, help="Contact distance weight")
    parser.add_argument("--w_fc", default=1.0, type=float, help="Force closure weight")
    parser.add_argument("--w_pen", default=100.0, type=float, help="Penetration weight")
    parser.add_argument("--w_spen", default=10.0, type=float, help="Self-penetration weight")
    parser.add_argument("--w_joints", default=1.0, type=float, help="Joint limits weight")
    parser.add_argument("--w_prior", default=0.0, type=float, help="Prior pose weight")
    parser.add_argument("--w_svd", default=0.1, type=float)
    # Trajectory-specific (disabled by default for sanity check)
    parser.add_argument("--w_ref", default=0.0, type=float, help="Reference tracking weight (trajectory only)")
    parser.add_argument("--w_smooth", default=0.0, type=float, help="Velocity smoothness weight (trajectory only)")

    # Optimizer settings - DEFAULT to mala to match sanity_check_optim.py
    parser.add_argument(
        "--optimizer",
        default="mala",
        type=str,
        choices=["adam", "mala"],
        help="Optimizer: 'mala' for grasp synthesis (matches sanity_check), 'adam' for refinement",
    )
    parser.add_argument("--lr", default=0.005, type=float, help="Learning rate for Adam")
    parser.add_argument(
        "--min_grad_norm", default=10.0, type=float, help="Minimum gradient norm to prevent vanishing (0=disabled)"
    )
    parser.add_argument("--switch_possibility", default=0.4, type=float, help="Contact switch probability")
    # MALA* settings (only used if --optimizer mala)
    parser.add_argument("--starting_temperature", default=18.0, type=float)
    parser.add_argument("--temperature_decay", default=0.95, type=float)
    parser.add_argument("--annealing_period", default=30, type=int)
    parser.add_argument("--step_size", default=0.005, type=float)
    parser.add_argument("--stepsize_period", default=50, type=int)
    parser.add_argument("--mu", default=0.98, type=float)
    parser.add_argument("--clip_grad", action="store_true")

    # Paths
    parser.add_argument("--data_root_path", default="./objects", type=str)
    parser.add_argument("--hand_name", default="allegro", type=str, choices=AVAILABLE_HANDS)
    parser.add_argument("--prior_file", default="configs/two_frame_prior.yaml", type=str)
    parser.add_argument("--mesh_extension", default=".obj", type=str)
    parser.add_argument("--grasp_type", default="all", type=str)
    parser.add_argument("--friction", default=0.2, type=float)
    parser.add_argument("--max_lambda_limit", default=20.0, type=float)
    parser.add_argument("--n_friction_cone", default=4, type=int)
    parser.add_argument("--energy_name", default="trajectory", type=str)

    # Output
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output_metrics", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)

    return parser.parse_args()


def load_trajectory_priors(
    prior_file: str,
    n_frames: int,
    batch_size: int,
    hand_model,
    device: torch.device,
) -> torch.Tensor:
    """
    Load priors and construct trajectory reference.

    If prior_file has exactly n_frames priors, use them as the trajectory.
    Otherwise, expand/interpolate to get n_frames.

    Returns:
        reference_trajectory: (B, T, D_hand) tensor
    """
    config = GraspPriorLoader.load_from_file(prior_file)

    n_priors = len(config.priors)
    D_hand = hand_model.n_dofs + 9  # 3 (trans) + 6 (rot6d) + n_dofs

    if n_priors == 0:
        raise ValueError(f"No priors found in {prior_file}")

    if n_priors >= n_frames:
        # Use first n_frames priors directly
        priors_to_use = config.priors[:n_frames]
    else:
        # Repeat/cycle priors to get n_frames
        priors_to_use = []
        for t in range(n_frames):
            priors_to_use.append(config.priors[t % n_priors])

    # Build trajectory for each frame
    trajectory = torch.zeros(batch_size, n_frames, D_hand, device=device)

    for t, prior in enumerate(priors_to_use):
        # Translation
        if prior.translation is not None:
            trans = torch.tensor(prior.translation, device=device, dtype=torch.float)
        else:
            trans = torch.zeros(3, device=device)

        # Rotation (quaternion w,x,y,z -> rotation matrix -> rot6d)
        if prior.rotation is not None:
            quat = torch.tensor(prior.rotation, device=device, dtype=torch.float)
            # Convert from (w,x,y,z) to (x,y,z,w) for roma
            quat_xyzw = quat[[1, 2, 3, 0]]
            R = roma.unitquat_to_rotmat(quat_xyzw.unsqueeze(0))[0]
        else:
            R = torch.eye(3, device=device)
        rot6d = R.T[:2].reshape(6)

        # Joints
        joints = hand_model.default_state.clone().to(device)
        if prior.joints is not None:
            for joint_name, angle in prior.joints.items():
                if joint_name in hand_model._actuated_joints_names:
                    idx = hand_model._actuated_joints_names.index(joint_name)
                    joints[idx] = angle

        # Combine
        frame_pose = torch.cat([trans, rot6d, joints])

        # Expand to batch with jitter for diversity
        for b in range(batch_size):
            jitter_trans = torch.randn(3, device=device) * config.jitter_translation
            jitter_rot = torch.randn(6, device=device) * config.jitter_rotation * 0.1
            jitter_joints = torch.randn_like(joints) * config.jitter_joints

            pose_with_jitter = frame_pose.clone()
            pose_with_jitter[:3] += jitter_trans
            pose_with_jitter[3:9] += jitter_rot
            pose_with_jitter[9:] = (joints + jitter_joints).clamp(hand_model.joints_lower, hand_model.joints_upper)

            trajectory[b, t] = pose_with_jitter

    return trajectory, config.prior_weight


def get_result_path(args, asset_id):
    """Get the result path for saving grasps."""
    if args.output_dir:
        path = args.output_dir
    else:
        path = os.path.join(
            args.data_root_path,
            args.object_code_list[asset_id],
            "grasp_predictions",
            args.hand_name,
            f"{args.n_contact}_contacts",
            args.energy_name,
        )
        if args.grasp_type in [None, "all"]:
            path = os.path.join(path, "default")
        else:
            path = os.path.join(path, args.grasp_type)
    os.makedirs(path, exist_ok=True)
    return path


def export_trajectory(
    args,
    hand_model,
    state: TrajectoryState,
    energies: torch.Tensor,
    object_model,
    cost_breakdown: Dict[str, torch.Tensor],
    suffix: str = "_trajectory",
):
    """
    Export trajectory in extended format compatible with visualize_result.py.

    Format follows docs/OPTIMIZATION_FRAMEWORK_DESIGN.md.
    """
    B, T, D_hand = state.hand_states.shape

    for asset_idx in range(len(args.object_code_list)):
        start_idx = asset_idx * args.batch_size
        end_idx = (asset_idx + 1) * args.batch_size

        # Get trajectories for this asset
        traj_hand = state.hand_states[start_idx:end_idx].detach().cpu()  # (B, T, D)
        traj_obj = state.object_states[start_idx:end_idx].detach().cpu()  # (B, T, 7)
        traj_energies = energies[start_idx:end_idx].detach().cpu()

        # === LEGACY FIELDS (last frame, for backward compat) ===
        last_frame = traj_hand[:, -1]  # (B, D_hand)

        # Convert rot6d to quaternion for legacy format
        hand_poses = robust_compute_rotation_matrix_from_ortho6d(last_frame[:, 3:9])
        hand_qxyzw = roma.rotmat_to_unitquat(hand_poses)
        hand_qwxyz = hand_qxyzw[:, [3, 0, 1, 2]]
        root_pose = torch.cat([last_frame[:, :3], hand_qwxyz], dim=1)
        joint_positions = last_frame[:, 9:]

        # Build legacy parameters dict
        parameters = {}
        for idx in range(joint_positions.shape[1]):
            parameters[hand_model._actuated_joints_names[idx]] = joint_positions[:, idx]
        parameters["root_pose"] = root_pose

        # === TRAJECTORY DATA ===
        trajectory_data = {
            "n_frames": T,
            "dt": state.dt,
            "hand_states": traj_hand,
            "object_states": traj_obj,
        }

        # === COST BREAKDOWN ===
        breakdown = {}
        for name, values in cost_breakdown.items():
            breakdown[name] = values[start_idx:end_idx].detach().cpu()

        # Get contact indices for this batch (per-trajectory, not expanded)
        # contact_point_indices is (B*T, n_contacts), we need (B, n_contacts)
        # Just take the first frame's contacts since they're all the same per trajectory
        n_contacts = hand_model.contact_point_indices.shape[-1]
        total_trajectories = hand_model.contact_point_indices.shape[0] // T
        batch_contacts = (
            hand_model.contact_point_indices.reshape(total_trajectories, T, n_contacts)[start_idx:end_idx, 0]
            .detach()
            .cpu()
        )

        # === FULL DATA DICT ===
        data = {
            # Legacy fields
            "values": traj_energies,
            "parameters": parameters,
            "contact_idx": batch_contacts,
            "grasp_type": args.grasp_type,
            "contact_links": hand_model._contact_links,
            # Trajectory data
            "trajectory": trajectory_data,
            # Per-frame energy (computed from last evaluation)
            "per_frame_energy": None,  # Would need per-frame evaluation
            # Cost breakdown
            "cost_breakdown": breakdown,
            # Metadata
            "metadata": {
                "optimizer": "MalaStarTrajectoryOptimizer",
                "n_iters": args.n_iter,
                "n_frames": T,
                "hand_name": args.hand_name,
                "prior_file": args.prior_file,
                "object_code": args.object_code_list[asset_idx],
                "data_root_path": args.data_root_path,
            },
        }

        # Save
        file_path = os.path.join(
            get_result_path(args, asset_idx),
            args.object_code_list[asset_idx] + f"{suffix}.dexgrasp.pt",
        )
        torch.save(data, file_path)
        print(f"\033[94m==> Exported trajectory to {os.path.abspath(file_path)}\033[0m")


def main():
    args = parse_args()

    # Set random seeds
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    np.seterr(all="raise")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profiler = get_profiler(enabled=args.profile, cuda_sync=True)

    num_objects = len(args.object_code_list)
    total_batch_size = num_objects * args.batch_size
    T = args.n_frames

    print("=" * 70)
    print(f"Hand-Object Trajectory Optimization (T={T})")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Objects: {args.object_code_list}")
    print(f"Batch size: {args.batch_size} × {num_objects} = {total_batch_size}")
    print(f"Frames: {T}")
    print(f"Iterations: {args.n_iter}")
    print("=" * 70)

    # =========================================================================
    # 1. Initialize hand and object models
    # =========================================================================
    hand_model = get_hand_model(args.hand_name, device, grasp_type=args.grasp_type)

    object_model = ObjectModel(
        data_root_path=args.data_root_path,
        batch_size_each=args.batch_size * T,  # Need space for all frames
        num_samples=1000,
        device=device,
    )
    object_model.initialize(args.object_code_list, extension=args.mesh_extension)

    print(f"\nHand: {args.hand_name}, n_dofs={hand_model.n_dofs}")
    print(f"n_contact_candidates: {hand_model.n_contact_candidates}")

    # =========================================================================
    # 2. Load trajectory priors
    # =========================================================================
    print(f"\nLoading trajectory priors from: {args.prior_file}")
    reference_hand, prior_weight = load_trajectory_priors(args.prior_file, T, total_batch_size, hand_model, device)
    print(f"  Loaded trajectory: shape={reference_hand.shape}")

    if args.w_prior == 0.0:
        args.w_prior = prior_weight
    print(f"  Prior weight: {args.w_prior}")

    # =========================================================================
    # 3. Create TrajectoryState
    # =========================================================================
    D_obj = 7  # pos(3) + quat(4)

    # Determine if we're in "sanity check" mode (no cross-frame costs)
    sanity_check_mode = args.w_ref == 0 and args.w_smooth == 0

    if sanity_check_mode:
        # SANITY CHECK MODE: Flatten to B*T independent samples with T=1
        # This makes the problem identical to sanity_check_optim.py with B*T batches
        flat_batch_size = total_batch_size * T
        object_at_origin = torch.zeros(flat_batch_size, 1, D_obj, device=device)
        object_at_origin[:, :, 6] = 1.0  # Identity quaternion (w=1)

        # Flatten reference: (B, T, D) -> (B*T, 1, D)
        flat_reference_hand = reference_hand.reshape(flat_batch_size, 1, -1)

        state = TrajectoryState(
            hand_states=flat_reference_hand.clone(),
            object_states=object_at_origin,
            dt=0.1,
        )
        print(f"\nSANITY CHECK MODE: Flattened to B*T={flat_batch_size} independent samples")
        print(f"TrajectoryState: B={state.B}, T={state.T}, D_hand={state.D_hand}")
    else:
        # TRAJECTORY MODE: Keep (B, T, D) structure
        object_at_origin = torch.zeros(total_batch_size, T, D_obj, device=device)
        object_at_origin[:, :, 6] = 1.0  # Identity quaternion (w=1)

        state = TrajectoryState(
            hand_states=reference_hand.clone(),
            object_states=object_at_origin,
            dt=0.1,
        )
        print(f"\nTRAJECTORY MODE: B={state.B}, T={state.T}")
        print(f"TrajectoryState: B={state.B}, T={state.T}, D_hand={state.D_hand}")

    # =========================================================================
    # 4. Initialize contact indices
    # =========================================================================
    if sanity_check_mode:
        # SANITY CHECK MODE: Each of the B*T samples has its own contacts
        flat_batch_size = total_batch_size * T
        initial_contacts = torch.randint(
            hand_model.n_contact_candidates,
            size=(flat_batch_size, args.n_contact),
            device=device,
        )
        # Flatten hand states for initialization: (B, T, D) -> (B*T, D)
        flat_reference = reference_hand.reshape(flat_batch_size, -1)

        # Set hand model with B*T samples
        hand_model.set_parameters(flat_reference, contact_point_indices=initial_contacts)
        print(f"  Hand model configured for {flat_batch_size} independent samples")
    else:
        # TRAJECTORY MODE: Same contacts for all frames in each trajectory
        # Shape: (B, n_contacts) - same contacts for all T frames in each trajectory
        initial_contacts_per_traj = torch.randint(
            hand_model.n_contact_candidates,
            size=(total_batch_size, args.n_contact),
            device=device,
        )
        # Expand to (B*T, n_contacts) for batched FK across all frames
        initial_contacts = (
            initial_contacts_per_traj.unsqueeze(1)
            .expand(total_batch_size, T, args.n_contact)
            .reshape(total_batch_size * T, args.n_contact)
        )
        # Flatten hand states for initialization: (B, T, D) -> (B*T, D)
        flat_reference = reference_hand.reshape(total_batch_size * T, -1)

        # Set hand model with all B*T samples
        hand_model.set_parameters(flat_reference, contact_point_indices=initial_contacts)
        print(f"  Hand model configured for {total_batch_size * T} samples (B*T)")

    # =========================================================================
    # 5. Create OptimizationContext
    # =========================================================================
    if sanity_check_mode:
        # SANITY CHECK MODE: Reference has B*T batches, T=1
        flat_batch_size = total_batch_size * T
        flat_reference_hand = reference_hand.reshape(flat_batch_size, 1, -1)
        reference = ReferenceTrajectory(
            hand_states=flat_reference_hand,
            object_states=object_at_origin,
            contact_fingers=None,
            n_contacts=args.n_contact,
            hand_type=args.hand_name,
            dt=0.1,
        )
    else:
        # TRAJECTORY MODE: Reference has B batches, T frames
        reference = ReferenceTrajectory(
            hand_states=reference_hand,
            object_states=object_at_origin,
            contact_fingers=None,
            n_contacts=args.n_contact,
            hand_type=args.hand_name,
            dt=0.1,
        )

    context = OptimizationContext(
        hand_model=hand_model,
        object_model=object_model,
        reference=reference,
        device=device,
        profiler=profiler,
    )

    # Set contact indices in context for cost evaluation
    context.set_contact_indices(initial_contacts)

    # =========================================================================
    # 6. Create OptimizationProblem
    # =========================================================================
    problem = OptimizationProblem(context, profiler=profiler)

    # -------------------------------------------------------------------------
    # CORE COSTS (same as sanity_check_optim.py for baseline comparison)
    # Each frame is treated INDEPENDENTLY - no cross-frame interaction
    # -------------------------------------------------------------------------
    problem.add_cost(
        ContactDistanceCost(
            name="contact_distance",
            weight=args.w_dis,
        )
    )

    fc_cost = ForceClosureCost(
        name="force_closure",
        weight=args.w_fc,
        config={"svd_gain": args.w_svd},
    )
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

    problem.add_cost(
        PenetrationCost(
            name="penetration",
            weight=args.w_pen,
        )
    )

    problem.add_cost(
        SelfPenetrationCost(
            name="self_penetration",
            weight=args.w_spen,
        )
    )

    problem.add_cost(
        JointLimitCost(
            name="joint_limits",
            weight=args.w_joints,
        )
    )

    # Prior pose cost (per-frame, same as sanity_check_optim.py)
    if args.w_prior > 0:
        prior_cost = PriorPoseCost(
            name="prior_pose",
            weight=args.w_prior,
        )
        if sanity_check_mode:
            # SANITY CHECK MODE: (B*T, D) - each sample has its own prior
            prior_flat = reference_hand.reshape(total_batch_size * T, -1)
        else:
            # TRAJECTORY MODE: (B*T, D) - flatten across frames
            prior_flat = reference_hand.reshape(total_batch_size * T, -1)
        prior_cost.set_prior_pose(prior_flat)
        problem.add_cost(prior_cost)

    # -------------------------------------------------------------------------
    # TRAJECTORY-SPECIFIC COSTS (disabled by default for sanity check)
    # -------------------------------------------------------------------------
    if args.w_ref > 0:
        problem.add_cost(
            ReferenceTrackingCost(
                name="reference_tracking",
                weight=args.w_ref,
                config={
                    "hand_weight": 1.0,
                    "object_weight": 0.0,
                    "hand_position_weight": 10.0,
                    "hand_rotation_weight": 5.0,
                    "finger_weight": 1.0,
                },
            )
        )

    if T > 1 and args.w_smooth > 0:
        problem.add_cost(
            VelocitySmoothnessCost(
                name="velocity_smoothness",
                weight=args.w_smooth,
            )
        )

    print("\nCosts configured:")
    for name, cost in problem.costs.items():
        print(f"  {name}: weight={cost.weight}")

    # =========================================================================
    # 7. Create Optimizer
    # =========================================================================
    # Determine if we're in "sanity check" mode (no cross-frame costs)
    sanity_check_mode = args.w_ref == 0 and args.w_smooth == 0

    if args.optimizer == "adam":
        optimizer = AdamOptimizer(
            lr=args.lr,
            min_grad_norm=args.min_grad_norm,
            debug=True,
        )
        print(f"\nOptimizer: AdamOptimizer")
        print(f"  lr={args.lr}, min_grad_norm={args.min_grad_norm}")
        # Initialize Adam with persistent parameters
        state = optimizer.initialize(state)
    elif sanity_check_mode:
        # SANITY CHECK MODE: Use MalaStarOptimizer (known working)
        # Treat B*T as independent samples, exactly like sanity_check_optim.py
        optimizer = MalaStarOptimizer(
            fix_contacts=False,  # Allow contact switching (like sanity_check)
            switch_possibility=args.switch_possibility,
            starting_temperature=args.starting_temperature,
            temperature_decay=args.temperature_decay,
            annealing_period=args.annealing_period,
            step_size=args.step_size,
            stepsize_period=args.stepsize_period,
            mu=args.mu,
            clip_grad=args.clip_grad,
            batch_size_per_object=args.batch_size * T,  # Total samples = B*T
            profiler=profiler,
        )
        print(f"\nOptimizer: MalaStarOptimizer (SANITY CHECK MODE - B*T independent samples)")
        print(f"  step_size={args.step_size}, temp={args.starting_temperature}")
        print(f"  switch_possibility={args.switch_possibility}")
        print(f"  Treating {total_batch_size}×{T} = {total_batch_size * T} as independent samples")
    else:
        # TRAJECTORY MODE: Use MalaStarTrajectoryOptimizer
        # Allow contact switching like in sanity check mode for better exploration
        optimizer = MalaStarTrajectoryOptimizer(
            fix_contacts=False,  # Allow contact switching for exploration
            switch_possibility=args.switch_possibility,
            starting_temperature=args.starting_temperature,
            temperature_decay=args.temperature_decay,
            annealing_period=args.annealing_period,
            step_size=args.step_size,
            stepsize_period=args.stepsize_period,
            mu=args.mu,
            clip_grad=args.clip_grad,
            batch_size_per_object=args.batch_size,
            profiler=profiler,
        )
        print(f"\nOptimizer: MalaStarTrajectoryOptimizer (TRAJECTORY MODE)")
        print(f"  step_size={args.step_size}, temp={args.starting_temperature}")
        print(f"  switch_possibility={args.switch_possibility} (contact switching enabled)")

    # =========================================================================
    # 8. Initial evaluation (no_grad to avoid graph conflict with optimizer)
    # =========================================================================
    with torch.no_grad():
        context.clear_step_cache()  # Clear cache before evaluation
        initial_costs = problem.evaluate_all(state)
        initial_energy = problem.total_energy(state)
    print(f"\nInitial energy: mean={initial_energy.mean():.2f}, best={initial_energy.min():.2f}")
    print("Initial cost breakdown:")
    for k, v in initial_costs.items():
        print(f"  {k}: mean={v.mean():.4f}")

    # =========================================================================
    # 9. Main optimization loop
    # =========================================================================
    print(f"\nStarting optimization...")
    start_time = time.perf_counter()

    # Track global best solution across all steps
    global_best_energy = float("inf")
    global_best_state = None
    global_best_contacts = None
    global_best_step = 0

    for step in tqdm(range(1, args.n_iter + 1), desc="Optimizing"):
        with profiler.section("step"):
            context.clear_step_cache()
            state = optimizer.step(state, problem)

            if step % 50 == 0:
                # Compute energy for logging
                with torch.no_grad():
                    context.clear_step_cache()
                    log_energy = problem.total_energy(state)

                current_best = log_energy.min().item()

                # Track global best (save both state AND contact indices)
                if current_best < global_best_energy:
                    global_best_energy = current_best
                    global_best_state = state.clone()
                    global_best_contacts = hand_model.contact_point_indices.clone()
                    global_best_step = step

                print(
                    f"  Step {step}: mean={log_energy.mean():.2f}, best={current_best:.2f}, global_best={global_best_energy:.2f} (step {global_best_step})"
                )

        profiler.step_done()

    total_time = time.perf_counter() - start_time

    # Use global best state for final output
    if global_best_state is not None:
        print(f"\nUsing global best from step {global_best_step} (energy={global_best_energy:.2f})")
        state = global_best_state
        # Restore contact indices
        context.set_contact_indices(global_best_contacts)
        flat_hand = state.hand_states.reshape(-1, state.D_hand)
        hand_model.set_parameters(flat_hand, contact_point_indices=global_best_contacts)

    # =========================================================================
    # 10. Final results
    # =========================================================================
    profiler.summary()

    with torch.no_grad():
        context.clear_step_cache()
        final_costs = problem.evaluate_all(state)
        final_energy = problem.total_energy(state)

    print("\n" + "=" * 70)
    print("=== Final Results ===")
    print("=" * 70)
    print(f"Energy: best={final_energy.min():.2f}, mean={final_energy.mean():.2f}")
    print("Breakdown:")
    for k, v in final_costs.items():
        print(f"  {k}: mean={v.mean():.4f}")
    print(f"\nTiming: {total_time:.2f}s, {total_time * 1000 / args.n_iter:.2f}ms/iter")

    # Export trajectory
    export_trajectory(args, hand_model, state, final_energy, object_model, final_costs)

    # =========================================================================
    # 11. Output metrics
    # =========================================================================
    if args.output_metrics:
        metrics = {
            "config": {
                "seed": args.seed,
                "n_iter": args.n_iter,
                "n_frames": T,
                "batch_size": args.batch_size,
                "object": args.object_code_list,
                "hand": args.hand_name,
            },
            "final": {
                "energy_mean": float(final_energy.mean()),
                "energy_min": float(final_energy.min()),
                "costs": {k: float(v.mean()) for k, v in final_costs.items()},
            },
            "timing": {
                "total_seconds": total_time,
                "per_iter_ms": total_time * 1000 / args.n_iter,
            },
        }

        os.makedirs(os.path.dirname(args.output_metrics) or ".", exist_ok=True)
        with open(args.output_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Metrics saved to {args.output_metrics}")

    print(f"\n✓ Trajectory optimization complete (T={T} frames)!")


if __name__ == "__main__":
    main()
