#!/usr/bin/env python3
"""
Extract grasp poses from saved checkpoints to use as priors.

Usage:
    python scripts/extract_prior_pose.py \
        --checkpoint ./objects/apple/grasp_predictions/allegro/8_contacts/graspqp/default/apple.dexgrasp.pt \
        --top_k 1 \
        --output configs/extracted_prior.yaml
"""

import argparse

import numpy as np
import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description="Extract grasp poses from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .dexgrasp.pt file")
    parser.add_argument("--top_k", type=int, default=1, help="Number of best grasps to extract")
    parser.add_argument("--output", type=str, default=None, help="Output YAML config file")
    parser.add_argument("--print_only", action="store_true", help="Only print, don't save")
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    data = torch.load(args.checkpoint, map_location="cpu")

    # Get energies and sort
    energies = data["values"]
    n_grasps = len(energies)
    sorted_indices = torch.argsort(energies)

    print(f"Found {n_grasps} grasps")
    print(f"Energy range: {energies.min():.3f} to {energies.max():.3f}")
    print(f"\nExtracting top {args.top_k} grasps:\n")

    # Extract parameters
    params = data["parameters"]
    root_pose = params["root_pose"]  # (N, 7) - [x, y, z, qw, qx, qy, qz]

    # Get joint names
    joint_names = [k for k in params.keys() if k != "root_pose"]
    joint_names_sorted = sorted(joint_names)  # Ensure consistent order

    priors = []
    for i in range(min(args.top_k, n_grasps)):
        idx = sorted_indices[i].item()
        energy = energies[idx].item()

        # Extract pose
        pose = root_pose[idx]  # [x, y, z, qw, qx, qy, qz]
        translation = pose[:3].tolist()
        quaternion = pose[3:7].tolist()  # [qw, qx, qy, qz]

        # Extract joint angles
        joints = {}
        for jname in joint_names_sorted:
            joints[jname] = params[jname][idx].item()

        print(f"Grasp {i+1} (index={idx}, energy={energy:.3f}):")
        print(f"  Translation: [{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}]")
        print(
            f"  Quaternion [w,x,y,z]: [{quaternion[0]:.4f}, {quaternion[1]:.4f}, {quaternion[2]:.4f}, {quaternion[3]:.4f}]"
        )
        print(f"  Joints: {len(joints)} values")
        print()

        # Build prior dict
        prior = {
            "translation": [round(x, 4) for x in translation],
            "quaternion": [round(x, 4) for x in quaternion],
            "joints": {k: round(v, 4) for k, v in joints.items()},
        }
        priors.append(prior)

    # Create config
    config = {
        "contact": {
            "mode": "uniform",
        },
        "priors": priors,
        "jitter_translation": 0.02,
        "jitter_rotation": 0.1,
        "jitter_joints": 0.1,
        "prior_weight": 10.0,
    }

    if args.print_only or args.output is None:
        print("=" * 60)
        print("YAML Config (copy to file):")
        print("=" * 60)
        print(yaml.dump(config, default_flow_style=None, sort_keys=False))
    else:
        with open(args.output, "w") as f:
            yaml.dump(config, f, default_flow_style=None, sort_keys=False)
        print(f"Saved config to: {args.output}")


if __name__ == "__main__":
    main()
