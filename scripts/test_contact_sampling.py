#!/usr/bin/env python3
"""
Interactive test script for HierarchicalContactSampler.

This script tests the contact sampling implementation with the actual hand model
and provides visual feedback via Plotly.

Usage:
    python scripts/test_contact_sampling.py --hand_name allegro --mode constrained
    python scripts/test_contact_sampling.py --hand_name allegro --mode guided --show_plotly
"""

import argparse
import os
import sys
from collections import Counter
from typing import Dict, List

import torch

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graspqp.core.contact_sampler import ContactSamplingConfig, HierarchicalContactSampler
from graspqp.hands import AVAILABLE_HANDS, get_hand_model


def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_link_structure(hand_model):
    """Print the contact structure of the hand model"""
    print_header("Hand Contact Structure")

    total_contacts = 0
    for link_name in hand_model.mesh:
        n_contacts = len(hand_model.mesh[link_name]["contact_candidates"])
        if n_contacts > 0:
            print(f"  {link_name}: {n_contacts} contact points")
            total_contacts += n_contacts

    print(f"\n  Total contact candidates: {total_contacts}")
    print(f"  hand_model.n_contact_candidates: {hand_model.n_contact_candidates}")


def analyze_samples(sampler, samples: torch.Tensor, config_name: str):
    """Analyze and print statistics about sampled contacts"""
    print_header(f"Sample Analysis: {config_name}")

    batch_size, n_contacts = samples.shape

    # Aggregate statistics across all batches
    link_counts = Counter()
    finger_counts = Counter()
    fingers_per_batch = []
    links_per_batch = []

    for b in range(batch_size):
        batch_links = set()
        batch_fingers = set()

        for idx in samples[b]:
            link_name = sampler._index_to_link_name(idx.item())
            finger = sampler.link_to_finger[link_name]

            link_counts[link_name] += 1
            finger_counts[finger] += 1
            batch_links.add(link_name)
            batch_fingers.add(finger)

        links_per_batch.append(len(batch_links))
        fingers_per_batch.append(len(batch_fingers))

    # Print finger distribution
    print("\n  Finger Distribution (across all samples):")
    total = sum(finger_counts.values())
    for finger, count in sorted(finger_counts.items()):
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {finger:10s}: {count:5d} ({pct:5.1f}%) {bar}")

    # Print link distribution
    print("\n  Link Distribution (top 10):")
    for link_name, count in link_counts.most_common(10):
        pct = count / total * 100
        print(f"    {link_name:20s}: {count:5d} ({pct:5.1f}%)")

    # Print diversity statistics
    print("\n  Diversity Statistics:")
    print(f"    Avg fingers per batch: {sum(fingers_per_batch) / len(fingers_per_batch):.2f}")
    print(f"    Min fingers per batch: {min(fingers_per_batch)}")
    print(f"    Max fingers per batch: {max(fingers_per_batch)}")
    print(f"    Avg links per batch:   {sum(links_per_batch) / len(links_per_batch):.2f}")


def test_constrained_validity(sampler, samples: torch.Tensor, allowed_links: List[str]):
    """Verify all samples are from allowed links"""
    print_header("Constrained Mode Validity Check")

    # Build allowed indices
    allowed_indices = set()
    for link_name in allowed_links:
        if link_name in sampler.link_to_indices:
            allowed_indices.update(sampler.link_to_indices[link_name].tolist())

    violations = 0
    total = samples.numel()

    for b in range(samples.shape[0]):
        for idx in samples[b]:
            if idx.item() not in allowed_indices:
                violations += 1
                link = sampler._index_to_link_name(idx.item())
                print(f"    VIOLATION: batch {b}, idx {idx.item()} -> {link}")

    if violations == 0:
        print(f"  ✓ All {total} samples are from allowed links")
    else:
        print(f"  ✗ {violations}/{total} samples violated constraints")

    return violations == 0


def test_finger_diversity(sampler, samples: torch.Tensor, min_fingers: int):
    """Verify minimum finger diversity constraint"""
    print_header(f"Finger Diversity Check (min={min_fingers})")

    violations = 0

    for b in range(samples.shape[0]):
        fingers = set()
        for idx in samples[b]:
            link_name = sampler._index_to_link_name(idx.item())
            fingers.add(sampler.link_to_finger[link_name])

        if len(fingers) < min_fingers:
            violations += 1
            print(f"    VIOLATION: batch {b} has only {len(fingers)} fingers: {fingers}")

    if violations == 0:
        print(f"  ✓ All {samples.shape[0]} batches have >= {min_fingers} fingers")
    else:
        print(f"  ✗ {violations}/{samples.shape[0]} batches violated constraint")

    return violations == 0


def visualize_contacts(hand_model, sampler, samples: torch.Tensor, batch_idx: int = 0):
    """Visualize sampled contacts using Plotly"""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        print("  Plotly not available, skipping visualization")
        return

    print_header(f"Visualizing Contacts (batch {batch_idx})")

    # Force browser renderer to avoid dumping HTML to terminal
    # Check PLOTLY_RENDERER env var first, then default to browser
    import os

    renderer = os.environ.get("PLOTLY_RENDERER", "browser")
    pio.renderers.default = renderer
    print(f"  Using Plotly renderer: {renderer}")

    # Set up hand model with dummy pose
    batch_size = samples.shape[0]
    n_dofs = hand_model.n_dofs

    # Create neutral pose
    hand_pose = torch.zeros(batch_size, 3 + 6 + n_dofs, device=hand_model.device)
    hand_pose[:, 3] = 1.0  # Identity rotation (first column of rot6d)
    hand_pose[:, 7] = 1.0  # Identity rotation (second column of rot6d)

    # Set parameters
    hand_model.set_parameters(hand_pose, samples)

    # Get visualization data
    data = hand_model.get_plotly_data(batch_idx, with_contact_points=True, opacity=0.7)

    # Highlight selected contact points
    contact_pts = hand_model.contact_points[batch_idx].detach().cpu().numpy()

    # Color code by finger
    finger_colors = {
        "index": "red",
        "middle": "green",
        "ring": "blue",
        "thumb": "orange",
        "pinky": "purple",
    }

    for i, idx in enumerate(samples[batch_idx]):
        link_name = sampler._index_to_link_name(idx.item())
        finger = sampler.link_to_finger[link_name]
        color = finger_colors.get(finger, "gray")

        data.append(
            go.Scatter3d(
                x=[contact_pts[i, 0]],
                y=[contact_pts[i, 1]],
                z=[contact_pts[i, 2]],
                mode="markers",
                marker=dict(size=10, color=color),
                name=f"{finger} - {link_name}",
                showlegend=(i < 8),  # Only show legend for first few
            )
        )

    fig = go.Figure(data=data)
    fig.update_layout(
        title=f"Contact Points (batch {batch_idx})",
        scene=dict(aspectmode="data"),
    )

    # Write to HTML file as fallback if browser doesn't work
    output_path = "/tmp/contact_sampling_vis.html"
    fig.write_html(output_path)
    print(f"  Saved visualization to: {output_path}")

    # Try to show in browser
    try:
        fig.show()
        print("  Visualization opened in browser")
    except Exception as e:
        print(f"  Could not open browser: {e}")
        print(f"  Open {output_path} manually to view")


def main():
    parser = argparse.ArgumentParser(description="Test HierarchicalContactSampler")
    parser.add_argument("--hand_name", type=str, default="allegro", choices=AVAILABLE_HANDS)
    parser.add_argument("--mode", type=str, default="all", choices=["uniform", "guided", "constrained", "all"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_contacts", type=int, default=12)
    parser.add_argument("--show_plotly", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load hand model
    print_header(f"Loading {args.hand_name} hand model")
    hand_model = get_hand_model(args.hand_name, device)
    print_link_structure(hand_model)

    # Define test configurations
    configs = {}

    if args.mode in ["uniform", "all"]:
        configs["uniform"] = ContactSamplingConfig(mode="uniform")

    if args.mode in ["guided", "all"]:
        # Get fingertip links
        fingertip_links = [link for link in hand_model.mesh.keys() if "link_3" in link or "tip" in link.lower()]
        configs["guided"] = ContactSamplingConfig(
            mode="guided",
            preferred_links=fingertip_links,
            preference_weight=0.8,
            min_fingers=2,
            max_contacts_per_link=2,
        )

    if args.mode in ["constrained", "all"]:
        fingertip_links = [link for link in hand_model.mesh.keys() if "link_3" in link or "tip" in link.lower()]
        configs["constrained"] = ContactSamplingConfig(
            mode="constrained",
            preferred_links=fingertip_links,
            min_fingers=2,
            max_contacts_per_link=2,
        )

    # Run tests for each configuration
    all_passed = True

    for config_name, config in configs.items():
        print_header(f"Testing {config_name.upper()} mode")
        print(f"  Config: {config}")

        # Create sampler
        sampler = HierarchicalContactSampler(hand_model, config)

        # Sample contacts
        samples = sampler.sample(args.batch_size, args.n_contacts)
        print(f"  Sampled shape: {samples.shape}")

        # Analyze samples
        analyze_samples(sampler, samples, config_name)

        # Run validity checks
        if config.mode == "constrained":
            passed = test_constrained_validity(sampler, samples, config.preferred_links)
            all_passed = all_passed and passed

        if config.min_fingers is not None:
            passed = test_finger_diversity(sampler, samples, config.min_fingers)
            all_passed = all_passed and passed

        # Optional visualization
        if args.show_plotly and config_name == args.mode:
            visualize_contacts(hand_model, sampler, samples, batch_idx=0)

    # Summary
    print_header("Test Summary")
    if all_passed:
        print("  ✓ All tests PASSED")
    else:
        print("  ✗ Some tests FAILED")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
