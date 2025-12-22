#!/usr/bin/env python
"""
System test for sanity_check_optim.py

This test ensures:
1. The sanity check script runs without errors
2. The optimization improves energy (correctness)
3. Individual energy components are within acceptable bounds (quality)
4. Results match the baseline within tolerance (regression detection)

Usage:
    # Default: verbose mode (streams optimization output live)
    python -m unittest tests.system.test_sanity_check -v

    # Quiet mode (captures output)
    VERBOSE=0 python -m unittest tests.system.test_sanity_check -v

To update the baseline after intentional changes:
    python scripts/update_baseline.py
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

# Get paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BASELINE_FILE = os.path.join(REPO_ROOT, "tests", "baselines", "sanity_check_baseline.json")

# Verbose mode is ON by default, use VERBOSE=0 to disable
VERBOSE = os.environ.get("VERBOSE", "1") == "1" or "--verbose" in sys.argv
if "--verbose" in sys.argv:
    sys.argv.remove("--verbose")


class TestSanityCheckOptim(unittest.TestCase):
    """System test for sanity check optimization."""

    # Quality thresholds for BEST grasp (min values)
    # Ensures at least ONE good grasp exists in the batch
    QUALITY_THRESHOLDS_MIN = {
        "E_dis": 0.05,  # Best grasp should have contact points very close to surface
        "E_fc": 2.0,  # Best grasp should have good force closure
        "E_pen": 0.01,  # Best grasp should have minimal penetration
        "E_spen": 0.01,  # Best grasp should have minimal self-penetration
        "E_prior": 0.2,  # Best grasp should track prior well (when w_prior=100)
    }

    # Quality thresholds for AVERAGE quality (mean values)
    # Ensures overall batch quality is reasonable
    QUALITY_THRESHOLDS_MEAN = {
        "E_dis": 0.3,  # Average contact distance should be reasonable
        "E_fc": 10.0,  # Average force closure
        "E_pen": 0.1,  # Average penetration should be low
        "E_spen": 0.1,  # Average self-penetration should be low
        "E_prior": 1.0,  # Average prior tracking
    }

    # Energy improvement ratio
    IMPROVEMENT_RATIO = 0.3  # final < initial * ratio

    # Baseline tolerance (10%)
    BASELINE_TOLERANCE = 0.10

    # Timing tolerance (20% - more lenient due to hardware variance)
    TIMING_TOLERANCE = 0.20

    # Batch size (larger batch = more stable results)
    BATCH_SIZE = 32

    @classmethod
    def setUpClass(cls):
        """Run the sanity check script once and capture results."""
        cls.results = None
        cls.run_error = None
        cls.exit_code = None

        # Create temp directory for outputs (avoids overwriting existing data)
        cls.temp_dir = tempfile.mkdtemp(prefix="graspqp_test_")
        cls.metrics_file = os.path.join(cls.temp_dir, "metrics.json")
        cls.output_dir = os.path.join(cls.temp_dir, "grasp_results")

        # Build command
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
            str(cls.BATCH_SIZE),
            "--n_iter",
            "500",
            "--prior_file",
            os.path.join(REPO_ROOT, "configs", "extracted_prior.yaml"),
            "--w_prior",
            "100",
            "--seed",
            "1",
            "--output_metrics",
            cls.metrics_file,
            "--output_dir",
            cls.output_dir,
        ]

        # Add profiling in verbose mode
        if VERBOSE:
            cmd.append("--profile")

        try:
            print(f"\n{'=' * 70}")
            print("Running sanity check script (this may take ~60 seconds)...")
            if VERBOSE:
                print("VERBOSE mode: streaming output live")
                print(f"Temp output dir: {cls.temp_dir}")
            print(f"{'=' * 70}")
            sys.stdout.flush()

            if VERBOSE:
                # Stream output live
                result = subprocess.run(
                    cmd,
                    timeout=300,  # 5 minute timeout
                )
                cls.exit_code = result.returncode
                cls.stdout = ""
                cls.stderr = ""
            else:
                # Capture output (quiet mode)
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )
                cls.exit_code = result.returncode
                cls.stdout = result.stdout
                cls.stderr = result.stderr

            if result.returncode != 0:
                cls.run_error = f"Script failed with exit code {result.returncode}"
                if cls.stderr:
                    cls.run_error += f"\nstderr: {cls.stderr}"
                print(f"ERROR: {cls.run_error}")
            else:
                # Load metrics
                if os.path.exists(cls.metrics_file):
                    with open(cls.metrics_file) as f:
                        cls.results = json.load(f)
                    print("✓ Script completed successfully, metrics loaded")
                else:
                    cls.run_error = f"Metrics file not created: {cls.metrics_file}"
                    print(f"ERROR: {cls.run_error}")

        except subprocess.TimeoutExpired:
            cls.run_error = "Script timed out after 5 minutes"
            print(f"ERROR: {cls.run_error}")
        except Exception as e:
            cls.run_error = f"Exception running script: {e}"
            print(f"ERROR: {cls.run_error}")

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files and directories."""
        import shutil

        if hasattr(cls, "temp_dir") and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_runs_without_error(self):
        """Test that the script runs without errors."""
        if self.run_error:
            self.fail(self.run_error)
        self.assertEqual(self.exit_code, 0, f"Script exited with code {self.exit_code}")
        self.assertIsNotNone(self.results, "Results not loaded")

    def test_energy_improves(self):
        """Test that optimization improves the total energy."""
        if not self.results:
            self.skipTest("Results not available")

        initial_energy = self.results["initial"]["total"]["mean"]
        final_energy = self.results["final"]["total"]["mean"]

        ratio = final_energy / initial_energy

        if VERBOSE:
            print(
                f"\n  Energy: {initial_energy:.2f} → {final_energy:.2f} (ratio={ratio:.2f}, threshold={self.IMPROVEMENT_RATIO})"
            )

        self.assertLess(
            final_energy,
            initial_energy,
            f"Energy should decrease: initial={initial_energy:.2f}, final={final_energy:.2f}",
        )
        self.assertLess(
            ratio,
            self.IMPROVEMENT_RATIO,
            f"Energy should improve significantly: ratio={ratio:.2f} should be < {self.IMPROVEMENT_RATIO}",
        )

    def test_contact_distance_quality(self):
        """Test contact distance quality (both best and average)."""
        if not self.results:
            self.skipTest("Results not available")

        E_dis_min = self.results["final"]["E_dis"]["min"]
        E_dis_mean = self.results["final"]["E_dis"]["mean"]
        thresh_min = self.QUALITY_THRESHOLDS_MIN["E_dis"]
        thresh_mean = self.QUALITY_THRESHOLDS_MEAN["E_dis"]

        if VERBOSE:
            print(f"\n  E_dis: min={E_dis_min:.4f} (thresh={thresh_min}), mean={E_dis_mean:.4f} (thresh={thresh_mean})")

        self.assertLess(E_dis_min, thresh_min, f"Best E_dis={E_dis_min:.4f} should be < {thresh_min}")
        self.assertLess(E_dis_mean, thresh_mean, f"Mean E_dis={E_dis_mean:.4f} should be < {thresh_mean}")

    def test_penetration_quality(self):
        """Test penetration quality (both best and average)."""
        if not self.results:
            self.skipTest("Results not available")

        E_pen_min = self.results["final"]["E_pen"]["min"]
        E_pen_mean = self.results["final"]["E_pen"]["mean"]
        thresh_min = self.QUALITY_THRESHOLDS_MIN["E_pen"]
        thresh_mean = self.QUALITY_THRESHOLDS_MEAN["E_pen"]

        if VERBOSE:
            print(f"\n  E_pen: min={E_pen_min:.4f} (thresh={thresh_min}), mean={E_pen_mean:.4f} (thresh={thresh_mean})")

        self.assertLess(E_pen_min, thresh_min, f"Best E_pen={E_pen_min:.4f} should be < {thresh_min}")
        self.assertLess(E_pen_mean, thresh_mean, f"Mean E_pen={E_pen_mean:.4f} should be < {thresh_mean}")

    def test_force_closure_quality(self):
        """Test force closure quality (both best and average)."""
        if not self.results:
            self.skipTest("Results not available")

        E_fc_min = self.results["final"]["E_fc"]["min"]
        E_fc_mean = self.results["final"]["E_fc"]["mean"]
        thresh_min = self.QUALITY_THRESHOLDS_MIN["E_fc"]
        thresh_mean = self.QUALITY_THRESHOLDS_MEAN["E_fc"]

        if VERBOSE:
            print(f"\n  E_fc: min={E_fc_min:.4f} (thresh={thresh_min}), mean={E_fc_mean:.4f} (thresh={thresh_mean})")

        self.assertLess(E_fc_min, thresh_min, f"Best E_fc={E_fc_min:.4f} should be < {thresh_min}")
        self.assertLess(E_fc_mean, thresh_mean, f"Mean E_fc={E_fc_mean:.4f} should be < {thresh_mean}")

    def test_self_penetration_quality(self):
        """Test self-penetration quality (both best and average)."""
        if not self.results:
            self.skipTest("Results not available")

        E_spen_min = self.results["final"]["E_spen"]["min"]
        E_spen_mean = self.results["final"]["E_spen"]["mean"]
        thresh_min = self.QUALITY_THRESHOLDS_MIN["E_spen"]
        thresh_mean = self.QUALITY_THRESHOLDS_MEAN["E_spen"]

        if VERBOSE:
            print(
                f"\n  E_spen: min={E_spen_min:.4f} (thresh={thresh_min}), mean={E_spen_mean:.4f} (thresh={thresh_mean})"
            )

        self.assertLess(E_spen_min, thresh_min, f"Best E_spen={E_spen_min:.4f} should be < {thresh_min}")
        self.assertLess(E_spen_mean, thresh_mean, f"Mean E_spen={E_spen_mean:.4f} should be < {thresh_mean}")

    def test_prior_tracking_quality(self):
        """Test prior tracking quality (both best and average)."""
        if not self.results:
            self.skipTest("Results not available")

        E_prior_min = self.results["final"]["E_prior"]["min"]
        E_prior_mean = self.results["final"]["E_prior"]["mean"]
        thresh_min = self.QUALITY_THRESHOLDS_MIN["E_prior"]
        thresh_mean = self.QUALITY_THRESHOLDS_MEAN["E_prior"]

        if VERBOSE:
            print(
                f"\n  E_prior: min={E_prior_min:.4f} (thresh={thresh_min}), mean={E_prior_mean:.4f} (thresh={thresh_mean})"
            )

        self.assertLess(E_prior_min, thresh_min, f"Best E_prior={E_prior_min:.4f} should be < {thresh_min}")
        self.assertLess(E_prior_mean, thresh_mean, f"Mean E_prior={E_prior_mean:.4f} should be < {thresh_mean}")

    def test_final_energies_match_baseline(self):
        """Test that final energies are within tolerance of baseline."""
        if not self.results:
            self.skipTest("Results not available")

        if not os.path.exists(BASELINE_FILE):
            self.skipTest(f"Baseline file not found: {BASELINE_FILE}")

        with open(BASELINE_FILE) as f:
            baseline = json.load(f)

        if VERBOSE:
            print(f"\n  Comparing against baseline (tolerance={self.BASELINE_TOLERANCE * 100}%):")

        # Compare final energy components
        for key in ["E_dis", "E_fc", "E_pen", "E_spen", "E_prior", "total"]:
            baseline_val = baseline["final"][key]["mean"]
            actual_val = self.results["final"][key]["mean"]

            if baseline_val == 0:
                # Handle zero baseline gracefully
                tolerance = 0.01
            else:
                tolerance = abs(baseline_val * self.BASELINE_TOLERANCE)

            diff = abs(actual_val - baseline_val)
            diff_pct = (diff / baseline_val * 100) if baseline_val != 0 else 0

            if VERBOSE:
                status = "✓" if diff < tolerance + 0.01 else "✗"
                print(f"    {status} {key}: {actual_val:.4f} (baseline={baseline_val:.4f}, diff={diff_pct:.1f}%)")

            self.assertLess(
                diff,
                tolerance + 0.01,  # Small epsilon for numerical stability
                f"{key}: actual={actual_val:.4f}, baseline={baseline_val:.4f}, "
                f"diff={diff_pct:.1f}% (tolerance={self.BASELINE_TOLERANCE * 100}%)",
            )

    def test_timing_not_regressed(self):
        """Test that timing hasn't regressed significantly from baseline."""
        if not self.results:
            self.skipTest("Results not available")

        if not os.path.exists(BASELINE_FILE):
            self.skipTest(f"Baseline file not found: {BASELINE_FILE}")

        with open(BASELINE_FILE) as f:
            baseline = json.load(f)

        baseline_time = baseline["timing"]["per_iter_ms"]
        actual_time = self.results["timing"]["per_iter_ms"]

        max_allowed = baseline_time * (1 + self.TIMING_TOLERANCE)

        # Only fail if significantly slower (allow for hardware variance)
        self.assertLess(
            actual_time,
            max_allowed,
            f"Timing regressed: {actual_time:.1f}ms/iter vs baseline {baseline_time:.1f}ms/iter "
            f"(tolerance: {self.TIMING_TOLERANCE * 100}%)",
        )

        # Also report timing info
        speedup = (baseline_time - actual_time) / baseline_time * 100
        if speedup > 5:
            print(f"\n  ✓ Timing improved by {speedup:.1f}%: {actual_time:.1f}ms vs {baseline_time:.1f}ms")
        elif speedup < -5:
            print(f"\n  ⚠ Timing regressed by {-speedup:.1f}%: {actual_time:.1f}ms vs {baseline_time:.1f}ms")


if __name__ == "__main__":
    unittest.main(verbosity=2)
