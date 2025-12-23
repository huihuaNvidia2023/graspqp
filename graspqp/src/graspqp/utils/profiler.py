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
    - Automatic hierarchical nesting (nested section() calls create parent.child names)
    - Optional CUDA synchronization for accurate GPU timing
    - Statistics: count, total, mean, std, min, max
    - Formatted summary output with tree structure

    Example:
        with profiler.section("step"):
            with profiler.section("energy"):       # becomes "step.energy"
                with profiler.section("costs"):    # becomes "step.energy.costs"
                    pass
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

        # Stack for automatic hierarchical naming
        self._section_stack: List[str] = []

    def _sync(self):
        """Synchronize CUDA if enabled."""
        if self.cuda_sync:
            torch.cuda.synchronize()

    def _get_full_name(self, name: str) -> str:
        """Get full hierarchical name based on current stack."""
        if self._section_stack:
            return ".".join(self._section_stack) + "." + name
        return name

    @contextmanager
    def section(self, name: str):
        """
        Context manager for timing a section.

        Automatically creates hierarchical names based on nesting.
        Nested sections are named "parent.child.grandchild".

        Args:
            name: Section name (will be prefixed with parent sections)

        Example:
            with profiler.section("step"):
                with profiler.section("energy"):  # recorded as "step.energy"
                    pass
        """
        if not self.enabled:
            yield
            return

        full_name = self._get_full_name(name)
        self._section_stack.append(name)

        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed = time.perf_counter() - start
            self._times[full_name].append(elapsed)
            self._section_stack.pop()

    def step_done(self):
        """Mark the end of an optimization step. Call this at the end of each iteration."""
        if not self.enabled:
            return
        self._step_count += 1

    def reset(self):
        """Reset all accumulated statistics."""
        self._times.clear()
        self._step_count = 0
        self._section_stack.clear()

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

    def summary(self, top_n: Optional[int] = None, debug: bool = False):
        """
        Print a formatted summary of all profiled sections.

        Args:
            top_n: If specified, only show top N sections by total time.
            debug: If True, print all raw section names before summary
        """
        if not self.enabled:
            print("Profiler was disabled.")
            return

        if not self._times:
            print("No profiling data collected.")
            return

        if debug:
            print("\n[DEBUG] All recorded sections:")
            for name, times in sorted(self._times.items()):
                print(f"  {name}: {len(times)} samples, total={sum(times)*1000:.2f}ms")
            print()

        import numpy as np

        # Collect stats for all sections
        stats = {}
        for name in self._times:
            stats[name] = self.get_stats(name)

        # Find the "step" section for percentage calculation, or use max
        total_ref = stats.get("step", {}).get("total", 0)
        if total_ref == 0:
            total_ref = max(s["total"] for s in stats.values()) if stats else 1

        # Build hierarchy tree for proper display ordering
        # Group children under parents, sort parents by time, then children by time
        all_sections = set(stats.keys())

        def get_existing_parent(name):
            """Find the nearest existing parent section."""
            if "." not in name:
                return None
            parent = name.rsplit(".", 1)[0]
            # Walk up until we find an existing parent or hit the top
            while parent and parent not in all_sections:
                if "." not in parent:
                    return None  # No existing parent found
                parent = parent.rsplit(".", 1)[0]
            return parent if parent in all_sections else None

        # Find top-level sections and their children
        top_level = []
        children = defaultdict(list)
        for name in stats.keys():
            parent = get_existing_parent(name)
            if parent is None:
                top_level.append(name)
            else:
                children[parent].append(name)

        # Sort top-level by time (descending)
        top_level.sort(key=lambda x: stats[x]["total"], reverse=True)

        # Recursively build display order with DFS, tracking tree level
        display_order = []  # List of (name, tree_level)

        def add_with_children(name, level):
            display_order.append((name, level))
            if name in children:
                sorted_kids = sorted(children[name], key=lambda x: stats[x]["total"], reverse=True)
                for kid in sorted_kids:
                    add_with_children(kid, level + 1)

        for name in top_level:
            add_with_children(name, 0)

        if top_n:
            display_order = display_order[:top_n]

        # Print header
        print(f"\n{'=' * 80}")
        print(f"PROFILING SUMMARY ({self._step_count} steps)")
        print(f"{'=' * 80}")
        print(
            f"{'Section':<35} {'Total(s)':>10} {'Mean(ms)':>10} {'Std(ms)':>9} {'Min(ms)':>9} {'Max(ms)':>9} {'%':>7}"
        )
        print(f"{'-' * 80}")

        # Print sections with tree-style indentation
        for i, (name, level) in enumerate(display_order):
            s = stats[name]
            short_name = name.split(".")[-1]

            # Build tree prefix based on tree level (not dot count)
            if level == 0:
                prefix = ""
            else:
                # Check if this is the last sibling at this level
                is_last = True
                for j in range(i + 1, len(display_order)):
                    future_name, future_level = display_order[j]
                    if future_level < level:
                        break  # Moved to parent's sibling
                    if future_level == level:
                        is_last = False  # Found a sibling
                        break
                prefix = "│  " * (level - 1) + ("└─ " if is_last else "├─ ")

            display_name = prefix + short_name
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
