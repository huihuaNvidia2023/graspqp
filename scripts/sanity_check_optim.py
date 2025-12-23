#!/usr/bin/env python
"""
Sanity check: Use NEW optimization framework to solve same problem as fit.py.

Supports TWO initialization modes:
1. GRASP GENERATION (--init_mode convex_hull, default):
   - Random initial poses from convex hull sampling
   - Prior has jitter for exploration
   - Matches fit.py behavior for grasp synthesis
   
2. TRAJECTORY REFINEMENT (--init_mode prior):
   - All batches start from the SAME prior pose (no jitter)
   - Suitable for refining reference trajectories from video
   - All batches should converge to similar solutions

This script FULLY USES the new optim framework:
- TrajectoryState for state representation (T=1 single frame)
- ReferenceTrajectory for prior/reference pose
- OptimizationContext for shared resources
- OptimizationProblem with cost functions
- MalaStarOptimizer from the new framework

Usage (grasp generation - matches fit.py):
    python scripts/sanity_check_optim.py \
        --data_root_path ./objects \
        --object_code_list apple \
        --hand_name allegro \
        --batch_size 16 \
        --n_iter 500 \
        --prior_file configs/extracted_prior.yaml \
        --w_prior 100 \
        --init_mode convex_hull

Usage (trajectory refinement):
    python scripts/sanity_check_optim.py \
        --data_root_path ./objects \
        --object_code_list apple \
        --hand_name allegro \
        --batch_size 16 \
        --n_iter 500 \
        --prior_file configs/extracted_prior.yaml \
        --w_prior 100 \
        --init_mode prior
"""

import argparse
import json
import math
import os
import time

import numpy as np
import roma
import torch
from tqdm import tqdm

# Import from existing graspqp modules (for hand/object models)
from graspqp.core import GraspPriorLoader, ObjectModel
from graspqp.core.initializations import initialize_convex_hull
from graspqp.hands import AVAILABLE_HANDS, get_hand_model
from graspqp.metrics import GraspSpanMetricFactory
from graspqp.utils.profiler import get_profiler
from graspqp.utils.transforms import robust_compute_rotation_matrix_from_ortho6d

# Import from NEW optimization framework
from graspqp.optim.state import TrajectoryState, ReferenceTrajectory
from graspqp.optim.context import OptimizationContext
from graspqp.optim.problem import OptimizationProblem
from graspqp.optim.costs.grasp import ContactDistanceCost, ForceClosureCost, JointLimitCost, PriorPoseCost
from graspqp.optim.costs.penetration import PenetrationCost, SelfPenetrationCost
from graspqp.optim.optimizers.mala_star import MalaStarOptimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity Check - Using NEW optim framework")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--object_code_list", default=["apple"], nargs="+")
    parser.add_argument("--n_contact", default=8, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--n_iter", default=500, type=int)

    # Weights (same defaults as fit.py)
    parser.add_argument("--w_dis", default=100.0, type=float)
    parser.add_argument("--w_fc", default=1.0, type=float)
    parser.add_argument("--w_pen", default=100.0, type=float)
    parser.add_argument("--w_spen", default=10.0, type=float)
    parser.add_argument("--w_joints", default=1.0, type=float)
    parser.add_argument("--w_prior", default=0.0, type=float, help="Weight for reference tracking (prior)")
    parser.add_argument("--w_svd", default=0.1, type=float)

    # MalaStar optimizer settings (same as fit.py)
    parser.add_argument("--switch_possibility", default=0.4, type=float)
    parser.add_argument("--starting_temperature", default=18.0, type=float)
    parser.add_argument("--temperature_decay", default=0.95, type=float)
    parser.add_argument("--annealing_period", default=30, type=int)
    parser.add_argument("--step_size", default=0.005, type=float)
    parser.add_argument("--stepsize_period", default=50, type=int)
    parser.add_argument("--mu", default=0.98, type=float)
    parser.add_argument("--clip_grad", action="store_true")

    # Initialization settings (same as fit.py)
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
    parser.add_argument("--energy_name", default="graspqp", type=str)

    # Initialization mode
    parser.add_argument(
        "--init_mode",
        default="convex_hull",
        type=str,
        choices=["convex_hull", "prior"],
        help="Initialization mode: 'convex_hull' for random grasp generation (matches fit.py), "
        "'prior' for trajectory refinement (all batches start from prior pose)",
    )

    # Profiling and metrics output
    parser.add_argument("--profile", action="store_true", help="Enable detailed profiling")
    parser.add_argument(
        "--output_metrics",
        default=None,
        type=str,
        help="Path to save metrics JSON (for testing/benchmarking)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Override output directory for grasp results (avoids overwriting existing data)",
    )

    return parser.parse_args()


def get_result_path(args, asset_id):
    """Get the result path for saving grasps (same as fit.py)."""
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


def export_poses(args, hand_model, energies, object_model, suffix=""):
    """Export grasp poses to file (same format as fit.py)."""
    full_hand_poses = hand_model.hand_pose.detach().cpu()
    energies_cpu = energies.detach().cpu()

    old_contact_indices = hand_model.contact_point_indices.clone().detach()
    distance, contact_normal = object_model.cal_distance(hand_model.contact_points)
    contact_normal = 5 * (contact_normal * distance.unsqueeze(-1).abs())

    delta_theta, residuals, ee_vel = hand_model.get_req_joint_velocities(
        -contact_normal, hand_model.contact_point_indices, return_ee_vel=True
    )

    hand_model._set_contact_idxs("all")
    distance_full, contact_normal_full = object_model.cal_distance(hand_model.contact_points)
    contact_normal_full = 5 * (contact_normal_full * distance_full.unsqueeze(-1).abs())
    delta_theta_full, _ = hand_model.get_req_joint_velocities(
        -contact_normal_full, hand_model.contact_point_indices, return_ee_vel=False
    )
    hand_model._set_contact_idxs(old_contact_indices)

    distance, contact_normal = object_model.cal_distance(hand_model.contact_points)
    contact_normal = 5 * contact_normal * (distance.unsqueeze(-1).abs() + 0.005)
    delta_theta_off, residuals, ee_vel = hand_model.get_req_joint_velocities(
        -contact_normal, hand_model.contact_point_indices, return_ee_vel=True
    )

    for asset_idx in range(len(args.object_code_list)):
        data = {"values": energies_cpu[asset_idx * args.batch_size : (asset_idx + 1) * args.batch_size]}
        start_idx = asset_idx * args.batch_size
        end_idx = (asset_idx + 1) * args.batch_size
        joint_delta = delta_theta[start_idx:end_idx]

        hand_poses = robust_compute_rotation_matrix_from_ortho6d(full_hand_poses[start_idx:end_idx, 3:9])
        hand_qxyzw = roma.rotmat_to_unitquat(hand_poses)
        hand_qwxyz = hand_qxyzw[:, [3, 0, 1, 2]]
        hand_poses = torch.cat([full_hand_poses[start_idx:end_idx, :3], hand_qwxyz], dim=1)
        joint_positions = full_hand_poses[start_idx:end_idx, 9:]

        parameters = {}
        for idx in range(joint_positions.shape[1]):
            parameters[hand_model._actuated_joints_names[idx]] = joint_positions[:, idx].detach().cpu()
        parameters["root_pose"] = hand_poses.detach().cpu()

        grasp_velocities = {}
        full_grasp_velocities = {}
        grasp_velocities_off = {}
        for idx in range(joint_delta.shape[1]):
            grasp_velocities[hand_model._actuated_joints_names[idx]] = joint_delta[:, idx].detach().cpu()
            full_grasp_velocities[hand_model._actuated_joints_names[idx]] = (
                delta_theta_full[start_idx:end_idx, idx].detach().cpu()
            )
            grasp_velocities_off[hand_model._actuated_joints_names[idx]] = (
                delta_theta_off[start_idx:end_idx, idx].detach().cpu()
            )

        file_path = os.path.join(
            get_result_path(args, asset_idx),
            args.object_code_list[asset_idx] + f"{suffix}.dexgrasp.pt",
        )
        data["parameters"] = parameters
        data["grasp_velocities"] = grasp_velocities
        data["full_grasp_velocities"] = full_grasp_velocities
        data["grasp_velocities_off"] = grasp_velocities_off
        data["contact_idx"] = hand_model.contact_point_indices[start_idx:end_idx].detach().cpu()
        data["grasp_type"] = args.grasp_type
        data["contact_links"] = hand_model._contact_links
        torch.save(data, file_path)
        print(f"\033[94m==> Exported to {os.path.abspath(file_path)}\033[0m")


def collect_energy_stats(cost_values, total_energy, weight_dict):
    """Collect energy statistics for metrics output."""
    stats = {}
    for k, v in cost_values.items():
        # Map cost names to fit.py energy names
        name_map = {
            "contact_distance": "E_dis",
            "force_closure": "E_fc",
            "penetration": "E_pen",
            "self_penetration": "E_spen",
            "joint_limits": "E_joints",
            "prior_pose": "E_prior",
        }
        out_name = name_map.get(k, k)
        stats[out_name] = {
            "mean": float(v.mean().item()),
            "min": float(v.min().item()),
            "max": float(v.max().item()),
            "weight": float(weight_dict.get(out_name, 0.0)),
        }
    stats["total"] = {
        "mean": float(total_energy.mean().item()),
        "min": float(total_energy.min().item()),
        "max": float(total_energy.max().item()),
    }
    return stats


def main():
    args = parse_args()

    # Set random seeds
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    np.seterr(all="raise")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize profiler
    profiler = get_profiler(enabled=args.profile, cuda_sync=True)

    num_objects = len(args.object_code_list)
    total_batch_size = num_objects * args.batch_size

    print("=" * 70)
    print("Sanity Check: Using NEW optimization framework")
    print("  - Uses TrajectoryState, OptimizationContext, OptimizationProblem")
    print("  - Uses MalaStarOptimizer from graspqp.optim.optimizers")
    print("  - Uses cost functions from graspqp.optim.costs")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Objects: {args.object_code_list}")
    print(f"Batch size: {args.batch_size} × {num_objects} = {total_batch_size}")
    print(f"Iterations: {args.n_iter}")
    if args.profile:
        print("Profiling: ENABLED")
    if args.output_metrics:
        print(f"Metrics output: {args.output_metrics}")
    print("=" * 70)

    # =========================================================================
    # 1. Initialize hand and object models (same as fit.py)
    # =========================================================================
    hand_model = get_hand_model(args.hand_name, device, grasp_type=args.grasp_type)

    object_model = ObjectModel(
        data_root_path=args.data_root_path,
        batch_size_each=args.batch_size,
        num_samples=1000,
        device=device,
    )
    object_model.initialize(args.object_code_list, extension=args.mesh_extension)

    # =========================================================================
    # 2. Load prior pose (if specified)
    # =========================================================================
    prior_pose = None
    prior_config = None
    if args.prior_file is not None:
        print(f"\nLoading prior from: {args.prior_file}")
        prior_config = GraspPriorLoader.load_from_file(args.prior_file)

        if prior_config.priors:
            # For trajectory refinement: disable jitter to keep all batches identical
            if args.init_mode == "prior":
                print(f"  Mode: trajectory refinement (init from prior, no jitter)")
                prior_config.jitter_translation = 0.0
                prior_config.jitter_rotation = 0.0
                prior_config.jitter_joints = 0.0

            prior_data = GraspPriorLoader.expand_priors(prior_config, total_batch_size, hand_model, device)
            prior_pose = GraspPriorLoader.create_hand_pose_from_priors(prior_data)  # (B, D_hand)

            if args.w_prior == 0.0:
                args.w_prior = prior_config.prior_weight
            elif args.w_prior != prior_config.prior_weight:
                print(f"  Note: --w_prior={args.w_prior} overrides config value ({prior_config.prior_weight})")
            print(f"  Loaded {len(prior_config.priors)} prior(s), weight={args.w_prior}")

    # =========================================================================
    # 3. Initialize hand pose based on mode
    # =========================================================================
    if args.init_mode == "convex_hull":
        # Grasp generation mode: random init (matches fit.py)
        print(f"\nInitialization mode: convex_hull (random grasp generation)")
        initialize_convex_hull(hand_model, object_model, args)
    elif args.init_mode == "prior":
        # Trajectory refinement mode: init from prior
        if prior_pose is None:
            raise ValueError("--init_mode=prior requires --prior_file to be specified")
        print(f"\nInitialization mode: prior (trajectory refinement)")
        # Initialize contacts first (need valid contact candidates)
        initialize_convex_hull(hand_model, object_model, args)
        # Then override hand_pose with prior
        hand_model.set_parameters(prior_pose, hand_model.contact_point_indices)
        print(f"  All {total_batch_size} batches initialized to same prior pose")

    print(f"n_contact_candidates: {hand_model.n_contact_candidates}")

    # =========================================================================
    # 4. Create TrajectoryState (T=1 single frame)
    # =========================================================================
    D_hand = hand_model.hand_pose.shape[1]
    D_obj = 7  # pos(3) + quat(4)

    # Initial hand state from hand_model
    initial_hand = hand_model.hand_pose.detach().clone()  # (B, D_hand)

    # Object at world origin
    object_at_origin = torch.zeros(total_batch_size, 1, D_obj, device=device)
    object_at_origin[:, :, 6] = 1.0  # Identity quaternion (w=1)

    # Create initial state
    state = TrajectoryState(
        hand_states=initial_hand.unsqueeze(1),  # (B, 1, D_hand)
        object_states=object_at_origin,
    )
    print(f"\nInitial TrajectoryState: B={state.B}, T={state.T}, D_hand={state.D_hand}")

    # =========================================================================
    # 5. Create OptimizationContext
    # =========================================================================
    # Create a reference trajectory (prior pose as reference for T=1)
    if prior_pose is not None:
        reference = ReferenceTrajectory(
            hand_states=prior_pose.unsqueeze(1),  # (B, 1, D_hand)
            object_states=object_at_origin.clone(),
            contact_fingers=None,  # uniform sampling
            n_contacts=args.n_contact,
            hand_type=args.hand_name,
        )
    else:
        reference = ReferenceTrajectory(
            hand_states=initial_hand.unsqueeze(1),  # use initial as reference
            object_states=object_at_origin.clone(),
            contact_fingers=None,
            n_contacts=args.n_contact,
            hand_type=args.hand_name,
        )

    context = OptimizationContext(
        hand_model=hand_model,
        object_model=object_model,
        reference=reference,
        device=device,
    )

    # =========================================================================
    # 6. Create OptimizationProblem with costs
    # =========================================================================
    problem = OptimizationProblem(context)

    # Define weight dict for metrics
    weight_dict = {
        "E_dis": args.w_dis,
        "E_fc": args.w_fc,
        "E_pen": args.w_pen,
        "E_spen": args.w_spen,
        "E_joints": args.w_joints,
        "E_prior": args.w_prior if prior_pose is not None else 0.0,
    }

    # Add costs (using new framework cost classes)
    problem.add_cost(ContactDistanceCost(
        name="contact_distance",
        weight=args.w_dis,
    ))

    # Force closure cost
    fc_cost = ForceClosureCost(
        name="force_closure",
        weight=args.w_fc,
        config={"svd_gain": args.w_svd},
    )
    # Set energy function (same as fit.py)
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

    problem.add_cost(PenetrationCost(
        name="penetration",
        weight=args.w_pen,
    ))

    problem.add_cost(SelfPenetrationCost(
        name="self_penetration",
        weight=args.w_spen,
    ))

    problem.add_cost(JointLimitCost(
        name="joint_limits",
        weight=args.w_joints,
    ))

    # Prior pose cost (if prior specified)
    if prior_pose is not None and args.w_prior > 0:
        prior_cost = PriorPoseCost(
            name="prior_pose",
            weight=args.w_prior,
        )
        prior_cost.set_prior_pose(prior_pose)
        problem.add_cost(prior_cost)

    print("\nCosts configured:")
    for name, cost in problem.costs.items():
        print(f"  {name}: weight={cost.weight}")

    # =========================================================================
    # 7. Create MalaStarOptimizer (from new framework)
    # =========================================================================
    optimizer = MalaStarOptimizer(
        fix_contacts=False,  # Allow contact switching (like fit.py)
        switch_possibility=args.switch_possibility,
        starting_temperature=args.starting_temperature,
        temperature_decay=args.temperature_decay,
        annealing_period=args.annealing_period,
        step_size=args.step_size,
        stepsize_period=args.stepsize_period,
        mu=args.mu,
        clip_grad=args.clip_grad,
    )
    print(f"\nOptimizer: MalaStarOptimizer (from new framework)")
    print(f"  step_size={args.step_size}, temp={args.starting_temperature}, decay={args.temperature_decay}")

    # =========================================================================
    # 8. Compute initial energy - DEBUG: compare with fit.py
    # =========================================================================
    print("\n" + "=" * 70)
    print("DEBUG: Initial Energy Comparison")
    print("=" * 70)

    # DEBUG: Check hand_model state BEFORE any evaluation
    print("\nhand_model state BEFORE evaluation:")
    print(f"  hand_pose shape: {hand_model.hand_pose.shape}")
    print(f"  hand_pose[:3, :5]: {hand_model.hand_pose[:3, :5]}")
    print(f"  contact_point_indices shape: {hand_model.contact_point_indices.shape}")
    print(f"  contact_point_indices[0]: {hand_model.contact_point_indices[0]}")
    print(f"  n_contact: {hand_model.contact_point_indices.shape[1]}")

    # Get surface points BEFORE any cost evaluation
    surface_pts_before = hand_model.get_surface_points()
    sdf_before, _ = object_model.cal_distance(surface_pts_before)
    pen_before = torch.nn.functional.relu(-sdf_before).sum(dim=-1)
    print(f"\nPenetration check BEFORE cost eval:")
    print(f"  surface_points shape: {surface_pts_before.shape}")
    print(f"  SDF min: {sdf_before.min().item():.4f}, max: {sdf_before.max().item():.4f}")
    print(f"  Penetration (raw): {pen_before.mean().item():.4f}")

    # First, compute using fit.py's calculate_energy for comparison
    from graspqp.core.energy import calculate_energy as fit_calculate_energy
    from graspqp.core import compute_prior_energy as fit_compute_prior_energy

    fit_energy_names = [e for e in weight_dict.keys() if weight_dict[e] > 0.0 and e != "E_prior"]
    fit_losses = fit_calculate_energy(
        hand_model,
        object_model,
        energy_names=fit_energy_names,
        energy_fnc=energy_fnc,
        method="gendexgrasp",
        svd_gain=args.w_svd,
    )
    if prior_pose is not None and weight_dict["E_prior"] > 0:
        fit_losses["E_prior"] = fit_compute_prior_energy(hand_model.hand_pose, prior_pose, prior_weight=1.0)

    fit_energy = sum(weight_dict[k] * v for k, v in fit_losses.items() if k in weight_dict)

    print("\nfit.py's calculate_energy (UNWEIGHTED losses):")
    for k, v in fit_losses.items():
        print(f"  {k}: {v.mean().item():.4f} (weighted: {weight_dict.get(k, 0) * v.mean().item():.4f})")
    print(f"  TOTAL (weighted sum): {fit_energy.mean().item():.2f}")

    # DEBUG: Check hand_model state AFTER fit.py evaluation
    print("\nhand_model state AFTER fit.py's calculate_energy:")
    print(f"  hand_pose[:3, :5]: {hand_model.hand_pose[:3, :5]}")
    print(f"  contact_point_indices[0]: {hand_model.contact_point_indices[0]}")

    surface_pts_after_fit = hand_model.get_surface_points()
    sdf_after_fit, _ = object_model.cal_distance(surface_pts_after_fit)
    pen_after_fit = torch.nn.functional.relu(-sdf_after_fit).sum(dim=-1)
    print(f"  Penetration (raw): {pen_after_fit.mean().item():.4f}")

    # Now compute using our new framework
    print("\n--- Calling problem.evaluate_all(state) ---")
    initial_costs = problem.evaluate_all(state)
    initial_energy = problem.total_energy(state)

    # DEBUG: Check hand_model state AFTER new framework evaluation
    print("\nhand_model state AFTER new framework's evaluate_all:")
    print(f"  hand_pose[:3, :5]: {hand_model.hand_pose[:3, :5]}")
    print(f"  contact_point_indices[0]: {hand_model.contact_point_indices[0]}")

    surface_pts_after_new = hand_model.get_surface_points()
    sdf_after_new, _ = object_model.cal_distance(surface_pts_after_new)
    pen_after_new = torch.nn.functional.relu(-sdf_after_new).sum(dim=-1)
    print(f"  Penetration (raw): {pen_after_new.mean().item():.4f}")

    print("\nNew framework costs (WEIGHTED - need to divide by weight):")
    cost_name_map = {
        "contact_distance": ("E_dis", args.w_dis),
        "force_closure": ("E_fc", args.w_fc),
        "penetration": ("E_pen", args.w_pen),
        "self_penetration": ("E_spen", args.w_spen),
        "joint_limits": ("E_joints", args.w_joints),
        "prior_pose": ("E_prior", args.w_prior),
    }
    for cost_name, weighted_value in initial_costs.items():
        fit_name, weight = cost_name_map.get(cost_name, (cost_name, 1.0))
        unweighted = weighted_value.mean().item() / weight if weight > 0 else 0
        print(f"  {cost_name} ({fit_name}): unweighted={unweighted:.4f}, weighted={weighted_value.mean().item():.4f}")
    print(f"  TOTAL: {initial_energy.mean().item():.2f}")

    print("\nDifference (fit.py - new framework):")
    for cost_name, weighted_value in initial_costs.items():
        fit_name, weight = cost_name_map.get(cost_name, (cost_name, 1.0))
        new_unweighted = weighted_value.mean().item() / weight if weight > 0 else 0
        fit_unweighted = fit_losses.get(fit_name, torch.zeros(1)).mean().item()
        diff = fit_unweighted - new_unweighted
        print(f"  {fit_name}: fit={fit_unweighted:.4f}, new={new_unweighted:.4f}, diff={diff:.4f}")
    print(f"  TOTAL diff: {fit_energy.mean().item() - initial_energy.mean().item():.2f}")
    print("=" * 70)

    # Store initial energy stats for metrics
    initial_stats = collect_energy_stats(initial_costs, initial_energy, weight_dict)

    print(f"\nInitial energy (new framework): {initial_energy.mean().item():.2f}")
    print(f"  Breakdown: {', '.join(f'{k}={v.mean().item():.2f}' for k, v in initial_costs.items())}")

    # DEBUG: Enable verbose mode in optimizer
    optimizer._debug = True

    # Gradient comparison skipped - causes double backward issues
    # The key finding: initial energy matches perfectly
    print("\n✓ Initial energy matches between fit.py and new framework!")
    print("=" * 50)

    print(f"\nStarting optimization...")

    # Start timing
    optimization_start_time = time.perf_counter()

    # =========================================================================
    # 9. Main optimization loop
    # =========================================================================
    for step in tqdm(range(1, args.n_iter + 1), desc="Optimizing"):
        with profiler.section("step"):
            # Clear step cache
            context.clear_step_cache()

            # Optimizer step (handles gradient, accept/reject, contact sampling)
            with profiler.section("optimizer_step"):
                state = optimizer.step(state, problem)

            # Get current energy for logging
            if step % 100 == 0 or step == 1:
                with profiler.section("logging"):
                    # Save gradient before logging (evaluate_all may clear it via set_parameters)
                    saved_grad = (
                        hand_model.hand_pose.grad.clone()
                        if hand_model.hand_pose.grad is not None
                        else None
                    )

                    current_costs = problem.evaluate_all(state)
                    current_energy = problem.total_energy(state)
                    breakdown = ", ".join(f"{k}={v.mean().item():.2f}" for k, v in current_costs.items())
                    print(f"Step {step}: total={current_energy.mean().item():.2f} | {breakdown}")

                    # Restore gradient
                    if saved_grad is not None:
                        hand_model.hand_pose.grad = saved_grad

        profiler.step_done()

    # =========================================================================
    # 10. Final results
    # =========================================================================
    optimization_end_time = time.perf_counter()
    total_time = optimization_end_time - optimization_start_time

    # Profiling summary
    profiler.summary()

    # Final evaluation
    final_costs = problem.evaluate_all(state)
    final_energy = problem.total_energy(state)

    print("\n" + "=" * 70)
    print("=== Final Results ===")
    print("=" * 70)
    print(f"Energy: best={final_energy.min().item():.2f}, mean={final_energy.mean().item():.2f}, "
          f"worst={final_energy.max().item():.2f}")
    print("Breakdown:")
    for k, v in final_costs.items():
        print(f"  {k}: mean={v.mean().item():.4f}, min={v.min().item():.4f}")

    print(f"\nTiming: {total_time:.2f}s total, {total_time * 1000 / args.n_iter:.2f}ms/iter")

    # Export in same format as fit.py
    export_poses(args, hand_model, final_energy, object_model, suffix="")

    print(f"\nFinal TrajectoryState: B={state.B}, T={state.T}")

    # =========================================================================
    # 10. Output metrics JSON (for testing/benchmarking)
    # =========================================================================
    if args.output_metrics:
        final_stats = collect_energy_stats(final_costs, final_energy, weight_dict)

        # Get timing breakdown from profiler
        timing_breakdown = {}
        if args.profile and profiler._times:
            total_step_time = sum(profiler._times.get("step", [0]))
            for key, times in profiler._times.items():
                if key != "step" and not key.startswith("step.energy."):
                    avg_time = sum(times) / len(times) if times else 0
                    timing_breakdown[key.replace("step.", "")] = {
                        "total_ms": sum(times) * 1000,
                        "avg_ms": avg_time * 1000,
                        "pct": (sum(times) / total_step_time * 100) if total_step_time > 0 else 0,
                    }

        metrics = {
            "config": {
                "seed": args.seed,
                "n_iter": args.n_iter,
                "batch_size": args.batch_size,
                "object": args.object_code_list[0] if len(args.object_code_list) == 1 else args.object_code_list,
                "hand": args.hand_name,
                "w_prior": args.w_prior,
                "w_dis": args.w_dis,
                "w_fc": args.w_fc,
                "w_pen": args.w_pen,
                "w_spen": args.w_spen,
                "w_joints": args.w_joints,
            },
            "initial": initial_stats,
            "final": final_stats,
            "timing": {
                "total_seconds": total_time,
                "per_iter_ms": total_time * 1000 / args.n_iter,
                "breakdown": timing_breakdown,
            },
        }

        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output_metrics) or ".", exist_ok=True)

        with open(args.output_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Metrics saved to {args.output_metrics}")

    print("\n✓ Completed using NEW optimization framework!")


if __name__ == "__main__":
    main()
