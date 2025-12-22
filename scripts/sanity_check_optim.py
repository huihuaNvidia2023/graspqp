#!/usr/bin/env python
"""
Sanity check script that mirrors fit.py exactly to verify the optimization works.

This uses the same MalaStar optimizer and energy calculation as fit.py.

Usage:
    python scripts/sanity_check_optim.py \
        --data_root_path ./objects \
        --object_code_list apple \
        --hand_name allegro \
        --batch_size 16 \
        --n_iter 500 \
        --prior_file configs/extracted_prior.yaml \
        --w_prior 100
"""

import argparse
import math
import os

import numpy as np
import roma
import torch
from tqdm import tqdm

# Import from graspqp (same imports as fit.py)
from graspqp.core import (
    ContactSamplingConfig,
    GraspPriorLoader,
    HierarchicalContactSampler,
    ObjectModel,
    compute_prior_energy,
)
from graspqp.core.energy import calculate_energy
from graspqp.core.initializations import initialize_convex_hull
from graspqp.core.optimizer import MalaStar  # Use the same optimizer as fit.py
from graspqp.hands import AVAILABLE_HANDS, get_hand_model
from graspqp.metrics import GraspSpanMetricFactory
from graspqp.utils.transforms import robust_compute_rotation_matrix_from_ortho6d


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity Check - mirrors fit.py")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--object_code_list", default=["apple"], nargs="+")
    parser.add_argument("--n_contact", default=8, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--n_iter", default=500, type=int)  # Same default as typical fit.py usage

    # Weights (same defaults as fit.py)
    parser.add_argument("--w_dis", default=100.0, type=float)
    parser.add_argument("--w_fc", default=1.0, type=float)
    parser.add_argument("--w_pen", default=100.0, type=float)
    parser.add_argument("--w_spen", default=10.0, type=float)
    parser.add_argument("--w_joints", default=1.0, type=float)
    parser.add_argument("--w_prior", default=0.0, type=float)
    parser.add_argument("--w_svd", default=0.1, type=float)

    # Optimizer hyperparameters (same as fit.py)
    parser.add_argument("--switch_possibility", default=0.4, type=float)
    parser.add_argument("--mu", default=0.98, type=float)
    parser.add_argument("--step_size", default=0.005, type=float)
    parser.add_argument("--stepsize_period", default=50, type=int)
    parser.add_argument("--starting_temperature", default=18, type=float)
    parser.add_argument("--annealing_period", default=30, type=int)
    parser.add_argument("--temperature_decay", default=0.95, type=float)
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

    # Reset settings
    parser.add_argument("--reset_epochs", default=600, type=int)
    parser.add_argument("--z_score_threshold", default=1.0, type=float)

    parser.add_argument("--data_root_path", default="./objects", type=str)
    parser.add_argument("--hand_name", default="allegro", type=str, choices=AVAILABLE_HANDS)
    parser.add_argument("--prior_file", default=None, type=str)
    parser.add_argument("--mesh_extension", default=".obj", type=str)
    parser.add_argument("--grasp_type", default="all", type=str)
    parser.add_argument("--friction", default=0.2, type=float)
    parser.add_argument("--max_lambda_limit", default=20.0, type=float)
    parser.add_argument("--n_friction_cone", default=4, type=int)
    parser.add_argument("--energy_name", default="graspqp", type=str)

    return parser.parse_args()


def get_result_path(args, asset_id):
    """Get the result path for saving grasps (same as fit.py)."""
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


def export_poses(args, hand_model, energy, object_model, suffix=""):
    """Export grasp poses to file (same as fit.py)."""
    full_hand_poses = hand_model.hand_pose.detach().cpu()
    energies = energy.detach().cpu()

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
        data = {"values": energies[asset_idx * args.batch_size : (asset_idx + 1) * args.batch_size]}
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


def main():
    args = parse_args()

    # Set random seeds exactly like fit.py (lines 173-175)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    np.seterr(all="raise")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_objects = len(args.object_code_list)
    total_batch_size = num_objects * args.batch_size

    print("=" * 50)
    print("Sanity Check - Mirrors fit.py Exactly")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Objects: {args.object_code_list}")
    print(f"Batch size: {args.batch_size} × {num_objects} = {total_batch_size}")
    print(f"Iterations: {args.n_iter}")
    print(f"Contacts: {args.n_contact}")
    print("=" * 50)

    # Initialize hand model (same as fit.py)
    hand_model = get_hand_model(args.hand_name, device, grasp_type=args.grasp_type)

    # Initialize object model (same as fit.py)
    object_model = ObjectModel(
        data_root_path=args.data_root_path,
        batch_size_each=args.batch_size,
        num_samples=1000,
        device=device,
    )
    object_model.initialize(args.object_code_list, extension=args.mesh_extension)

    # Initialize with convex hull (same as fit.py)
    initialize_convex_hull(hand_model, object_model, args)

    print("n_contact_candidates", hand_model.n_contact_candidates)
    print("total batch size", total_batch_size)

    # Optimizer config (same as fit.py)
    optim_config = {
        "switch_possibility": args.switch_possibility,
        "starting_temperature": args.starting_temperature,
        "temperature_decay": args.temperature_decay,
        "annealing_period": args.annealing_period,
        "step_size": args.step_size,
        "stepsize_period": args.stepsize_period,
        "mu": args.mu,
        "device": device,
        "batch_size": args.batch_size,
        "clip_grad": args.clip_grad,
    }

    # Setup prior & contact sampling (same as fit.py lines 352-401)
    contact_sampler = None
    prior_pose = None

    if args.prior_file is not None:
        print(f"Loading config from: {args.prior_file}")
        prior_config = GraspPriorLoader.load_from_file(args.prior_file)

        # Setup contact sampler from config
        contact_cfg = prior_config.contact
        if contact_cfg.mode != "uniform" or contact_cfg.links is not None:
            contact_config = ContactSamplingConfig(
                mode=contact_cfg.mode,
                preferred_links=contact_cfg.links,
                preference_weight=contact_cfg.preference_weight,
                min_fingers=contact_cfg.min_fingers,
                max_contacts_per_link=contact_cfg.max_contacts_per_link,
            )
            contact_sampler = HierarchicalContactSampler(hand_model, contact_config)
            print(f"  Contact sampling: mode={contact_cfg.mode}, links={contact_cfg.links}")

        # Setup prior poses if any priors defined
        if prior_config.priors:
            prior_data = GraspPriorLoader.expand_priors(prior_config, total_batch_size, hand_model, device)
            prior_pose = GraspPriorLoader.create_hand_pose_from_priors(prior_data)
            # Use command-line --w_prior if set (non-zero), otherwise use config file value
            if args.w_prior == 0.0:
                args.w_prior = prior_config.prior_weight
            elif args.w_prior != prior_config.prior_weight:
                print(f"  Note: --w_prior={args.w_prior} overrides config file value ({prior_config.prior_weight})")
            print(f"  Loaded {len(prior_config.priors)} prior(s), weight={args.w_prior}")

            # Override with per-batch contact configs if priors specify them (same as fit.py lines 384-392)
            if prior_data.get("contact_configs"):
                has_per_batch = any(cfg.mode != "uniform" for cfg in prior_data["contact_configs"])
                if has_per_batch:
                    contact_samplers = [
                        HierarchicalContactSampler(hand_model, cfg) for cfg in prior_data["contact_configs"]
                    ]
                    optim_config["contact_samplers"] = contact_samplers
                    contact_sampler = None  # Use per-batch samplers instead
                    print("  Using per-batch contact configurations")

    # Add contact sampler to optimizer config
    if contact_sampler is not None:
        optim_config["contact_sampler"] = contact_sampler

    if prior_pose is not None:
        optim_config["prior_pose"] = prior_pose
        optim_config["prior_weight"] = args.w_prior

    # Create MalaStar optimizer (same as fit.py line 403)
    optimizer = MalaStar(hand_model, **optim_config)

    # Create energy function (same as fit.py lines 416-424)
    energy_fnc = GraspSpanMetricFactory.create(
        GraspSpanMetricFactory.MetricType.GRASPQP,
        solver_kwargs={
            "friction": args.friction,
            "max_limit": args.max_lambda_limit,
            "n_cone_vecs": args.n_friction_cone,
        },
    )

    # Weight dictionary (same as fit.py lines 431-439)
    weight_dict = {
        "E_dis": args.w_dis,
        "E_fc": args.w_fc,
        "E_pen": args.w_pen,
        "E_spen": args.w_spen,
        "E_joints": args.w_joints,
        "E_prior": args.w_prior if prior_pose is not None else 0.0,
    }

    energy_names = [e for e in weight_dict.keys() if weight_dict[e] > 0.0 and e != "E_prior"]
    energy_kwargs = {"method": "gendexgrasp", "svd_gain": args.w_svd}

    print(f"Energy weights: {weight_dict}")
    print(f"Active energies: {energy_names}")

    # Initial forward/backward pass (same as fit.py lines 455-474)
    losses = calculate_energy(
        hand_model,
        object_model,
        energy_names=energy_names,
        energy_fnc=energy_fnc,
        **energy_kwargs,
    )

    # Add prior energy if configured
    if prior_pose is not None and weight_dict["E_prior"] > 0:
        losses["E_prior"] = compute_prior_energy(hand_model.hand_pose, prior_pose, prior_weight=1.0)

    # Accumulate energy exactly like fit.py (lines 467-471)
    energy = 0
    for loss_name, loss_value in losses.items():
        if loss_name not in weight_dict:
            raise ValueError(f"Loss name {loss_name} not in weight_dict")
        energy += weight_dict[loss_name] * loss_value

    energy.sum().backward()
    optimizer.zero_grad()

    print(f"\nStarting optimization (MalaStar)...")

    # Main optimization loop (same structure as fit.py lines 476-578)
    for step in tqdm(range(1, args.n_iter + 1), desc="Optimizing"):
        # try_step (same as fit.py line 479)
        s = optimizer.try_step()
        reset_mask = None

        # Reset logic (same as fit.py lines 482-505)
        E_fc_batch = energy.view(-1, args.batch_size)
        mean = E_fc_batch.mean(-1)
        std = E_fc_batch.std(-1)
        z_score = ((E_fc_batch - mean.unsqueeze(-1)) / std.unsqueeze(-1)).view(-1)

        if (
            args.reset_epochs is not None
            and step % args.reset_epochs == 0
            and (step < args.n_iter - 2 * args.reset_epochs)
        ):
            reset_mask = z_score > args.z_score_threshold
            if reset_mask.sum() > 0:
                print(f"Resetting {reset_mask.sum()} envs")
                initialize_convex_hull(hand_model, object_model, args, env_mask=reset_mask)
                optimizer.reset_envs(reset_mask)

        optimizer.zero_grad()

        # Calculate new energies (same as fit.py lines 510-517)
        new_energies = calculate_energy(
            hand_model,
            object_model,
            energy_fnc=energy_fnc,
            energy_names=energy_names,
            **energy_kwargs,
        )

        # Add prior energy if configured
        if prior_pose is not None and weight_dict["E_prior"] > 0:
            new_energies["E_prior"] = compute_prior_energy(hand_model.hand_pose, prior_pose, prior_weight=1.0)

        # Accumulate new_energy exactly like fit.py (lines 523-527)
        new_energy = 0
        for loss_name, loss_value in new_energies.items():
            if loss_name not in weight_dict:
                raise ValueError(f"Loss name {loss_name} not in weight_dict")
            new_energy += weight_dict[loss_name] * loss_value

        # Backward pass (same as fit.py line 530)
        new_energy.sum().backward()

        # Accept step (same as fit.py lines 536-549)
        with torch.no_grad():
            accept, t = optimizer.accept_step(
                energy,
                new_energy,
                reset_mask,
                z_score,
                args.z_score_threshold,
            )

            energy[accept] = new_energy[accept]
            for loss_name, loss_value in new_energies.items():
                if loss_name not in weight_dict:
                    raise ValueError(f"Loss name {loss_name} not in weight_dict")
                losses[loss_name][accept] = loss_value[accept]

        # Logging
        if step % 50 == 0 or step == 1:
            loss_str = ", ".join(f"{k}={v.mean().item():.3f}" for k, v in losses.items())
            print(f"Step {step}: total={energy.mean().item():.3f} | {loss_str}")

    # Final results
    print("\n" + "=" * 50)
    print("=== Final Results ===")
    print("=" * 50)
    print(f"Energy: best={energy.min().item():.2f}, mean={energy.mean().item():.2f}, worst={energy.max().item():.2f}")
    print("Breakdown:")
    for k, v in losses.items():
        print(f"  {k}: mean={v.mean().item():.4f}, min={v.min().item():.4f}")

    # Export results (same format as fit.py)
    export_poses(args, hand_model, energy, object_model, suffix="")

    print("\n✓ Optimization completed successfully!")
    print("This sanity check uses the same MalaStar optimizer and energy calculation as fit.py")


if __name__ == "__main__":
    main()
