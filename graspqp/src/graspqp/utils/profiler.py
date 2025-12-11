"""
Modular profiler for optimization loops.

Provides hierarchical timing with minimal overhead when disabled.

Usage:
    profiler = StepProfiler(enabled=True, cuda_sync=True)

    for step in range(n_iter):
        with profiler.section("try_step"):
            optimizer.try_step()

        with profiler.section("energy"):
            calculate_energy(...)

        profiler.step_done()

    profiler.summary()
"""

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch


class StepProfiler:
    """
    Lightweight profiler for optimization loops with hierarchical section timing.

    Features:
    - Enable/disable with zero overhead when disabled
    - Hierarchical sections via dot-notation (e.g., "energy.sdf_contact")
    - Optional CUDA synchronization for accurate GPU timing
    - Statistics: count, total, mean, std, min, max
    - Formatted summary output with percentages
    """

    def __init__(self, enabled: bool = False, cuda_sync: bool = True):
        """
        Initialize the profiler.

        Args:
            enabled: Whether profiling is active. When False, all operations are no-ops.
            cuda_sync: Whether to call torch.cuda.synchronize() before timing.
                       Required for accurate GPU timing but adds slight overhead.
        """
        self.enabled = enabled
        self.cuda_sync = cuda_sync and torch.cuda.is_available()

        # Storage: section_name -> list of durations
        self._times: Dict[str, List[float]] = defaultdict(list)
        self._step_count = 0

        # For tracking current step's total time
        self._step_start: Optional[float] = None

    def _sync(self):
        """Synchronize CUDA if enabled."""
        if self.cuda_sync:
            torch.cuda.synchronize()

    @contextmanager
    def section(self, name: str):
        """
        Context manager for timing a section.

        Args:
            name: Section name. Use dot-notation for hierarchy (e.g., "energy.qp_solver")

        Example:
            with profiler.section("energy.sdf_contact"):
                distance = object_model.cal_distance(points)
        """
        if not self.enabled:
            yield
            return

        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed = time.perf_counter() - start
            self._times[name].append(elapsed)

    def step_done(self):
        """Mark the end of an optimization step. Call this at the end of each iteration."""
        if not self.enabled:
            return
        self._step_count += 1

    def reset(self):
        """Reset all accumulated statistics."""
        self._times.clear()
        self._step_count = 0

    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a section.

        Returns:
            Dict with keys: count, total, mean, std, min, max
        """
        times = self._times.get(name, [])
        if not times:
            return {"count": 0, "total": 0, "mean": 0, "std": 0, "min": 0, "max": 0}

        import numpy as np

        arr = np.array(times)
        return {
            "count": len(arr),
            "total": arr.sum(),
            "mean": arr.mean(),
            "std": arr.std(),
            "min": arr.min(),
            "max": arr.max(),
        }

    def summary(self, top_n: Optional[int] = None):
        """
        Print a formatted summary of all profiled sections.

        Args:
            top_n: If specified, only show top N sections by total time.
        """
        if not self.enabled:
            print("Profiler was disabled.")
            return

        if not self._times:
            print("No profiling data collected.")
            return

        import numpy as np

        # Collect stats for all sections
        stats = {}
        for name in self._times:
            stats[name] = self.get_stats(name)

        # Sort by total time (descending)
        sorted_names = sorted(stats.keys(), key=lambda x: stats[x]["total"], reverse=True)

        if top_n:
            sorted_names = sorted_names[:top_n]

        # Find the "step" section for percentage calculation, or use max
        total_ref = stats.get("step", {}).get("total", 0)
        if total_ref == 0:
            total_ref = max(s["total"] for s in stats.values()) if stats else 1

        # Print header
        print(f"\n{'=' * 80}")
        print(f"PROFILING SUMMARY ({self._step_count} steps)")
        print(f"{'=' * 80}")
        print(
            f"{'Section':<35} {'Total(s)':>10} {'Mean(ms)':>10} {'Std(ms)':>9} {'Min(ms)':>9} {'Max(ms)':>9} {'%':>7}"
        )
        print(f"{'-' * 80}")

        # Build hierarchy for display
        for name in sorted_names:
            s = stats[name]
            # Indentation based on dots
            depth = name.count(".")
            prefix = "  " * depth + ("└─ " if depth > 0 else "")
            display_name = prefix + name.split(".")[-1]

            pct = (s["total"] / total_ref * 100) if total_ref > 0 else 0

            print(
                f"{display_name:<35} "
                f"{s['total']:>10.3f} "
                f"{s['mean']*1000:>10.2f} "
                f"{s['std']*1000:>9.2f} "
                f"{s['min']*1000:>9.2f} "
                f"{s['max']*1000:>9.2f} "
                f"{pct:>6.1f}%"
            )

        print(f"{'=' * 80}\n")

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Export all statistics as a dictionary."""
        return {name: self.get_stats(name) for name in self._times}


# Convenience: a disabled profiler singleton for when profiling is off
_disabled_profiler = StepProfiler(enabled=False)


def get_profiler(enabled: bool = False, cuda_sync: bool = True) -> StepProfiler:
    """
    Factory function to get a profiler instance.

    Returns a disabled singleton when enabled=False for zero overhead.
    """
    if not enabled:
        return _disabled_profiler
    return StepProfiler(enabled=True, cuda_sync=cuda_sync)
