#!/usr/bin/env python
"""
Update the baseline for system tests.

Run this after making intentional changes that affect optimization results.

Usage:
    python scripts/update_baseline.py

This will:
1. Run sanity_check_optim.py with the standard test configuration
2. Save the metrics to tests/baselines/sanity_check_baseline.json
3. Print a summary of the changes
"""

import json
import os
import subprocess
import sys

# Get paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
BASELINE_FILE = os.path.join(REPO_ROOT, "tests", "baselines", "sanity_check_baseline.json")


def main():
    # Load existing baseline if present
    old_baseline = None
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE) as f:
            old_baseline = json.load(f)

    # Run sanity check with profile to get timing breakdown
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "scripts", "sanity_check_optim.py"),
        "--data_root_path",
        os.path.join(REPO_ROOT, "objects"),
        "--object_code_list",
        "apple",
        "--hand_name",
        "allegro",
        "--batch_size",
        "16",
        "--n_iter",
        "500",
        "--prior_file",
        os.path.join(REPO_ROOT, "configs", "extracted_prior.yaml"),
        "--w_prior",
        "100",
        "--seed",
        "1",
        "--profile",
        "--output_metrics",
        BASELINE_FILE,
    ]

    print("=" * 70)
    print("Updating baseline...")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ Script failed with exit code {result.returncode}")
        sys.exit(1)

    # Load new baseline
    with open(BASELINE_FILE) as f:
        new_baseline = json.load(f)

    # Print summary
    print("\n" + "=" * 70)
    print("Baseline Updated!")
    print("=" * 70)

    if old_baseline:
        print("\nChanges from previous baseline:")
        print("-" * 50)

        # Compare final energies
        for key in ["E_dis", "E_fc", "E_pen", "E_spen", "E_prior", "total"]:
            old_val = old_baseline["final"][key]["mean"]
            new_val = new_baseline["final"][key]["mean"]
            if old_val != 0:
                change = (new_val - old_val) / old_val * 100
                direction = "↑" if change > 0 else "↓" if change < 0 else "="
                print(f"  {key}: {old_val:.4f} → {new_val:.4f} ({direction} {abs(change):.1f}%)")
            else:
                print(f"  {key}: {old_val:.4f} → {new_val:.4f}")

        # Compare timing
        old_time = old_baseline["timing"]["per_iter_ms"]
        new_time = new_baseline["timing"]["per_iter_ms"]
        change = (new_time - old_time) / old_time * 100
        direction = "↑ slower" if change > 0 else "↓ faster" if change < 0 else "="
        print(f"\n  Timing: {old_time:.1f}ms → {new_time:.1f}ms ({direction} {abs(change):.1f}%)")
    else:
        print("\nNew baseline created (no previous baseline to compare)")

    print(f"\n✓ Baseline saved to: {BASELINE_FILE}")


if __name__ == "__main__":
    main()
