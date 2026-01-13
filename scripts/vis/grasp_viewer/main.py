#!/usr/bin/env python3
"""CLI entry point for grasp trajectory visualizer.

Usage:
    # View optimized trajectories
    python scripts/vis/grasp_viewer/main.py --optimized path/to/optimized.dexgrasp.pt

    # View with reference comparison
    python scripts/vis/grasp_viewer/main.py \\
        --optimized path/to/optimized.dexgrasp.pt \\
        --reference path/to/reference.yaml

    # View reference only
    python scripts/vis/grasp_viewer/main.py --reference path/to/reference.yaml
"""

import argparse
import sys
from pathlib import Path

# Add parent directories to path for direct execution
# IMPORTANT: Handle ROS scripts package collision by ensuring repo root comes first
_THIS_DIR = Path(__file__).parent
_REPO_ROOT = _THIS_DIR.parent.parent.parent

# Remove repo root if it exists, then re-add at position 0 to ensure priority
_repo_str = str(_REPO_ROOT)
sys.path = [p for p in sys.path if p != _repo_str]
sys.path.insert(0, _repo_str)

# Also need to clear any cached 'scripts' module that ROS may have loaded
if 'scripts' in sys.modules:
    del sys.modules['scripts']


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Grasp Trajectory Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--optimized",
        "-o",
        type=str,
        default=None,
        help="Path to optimized trajectory file (.dexgrasp.pt)",
    )

    parser.add_argument(
        "--reference",
        "-r",
        type=str,
        default=None,
        help="Path to reference trajectory file (.yaml)",
    )

    parser.add_argument(
        "--asset-dir",
        type=str,
        default=None,
        help="Path to graspqp assets directory (default: auto-detect)",
    )

    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8080,
        help="Port for viser server (default: 8080)",
    )

    parser.add_argument(
        "--view-mode",
        type=str,
        choices=["playback", "ghost"],
        default="playback",
        help="Initial view mode (default: playback)",
    )

    args = parser.parse_args()

    # Validate inputs
    if args.optimized is None and args.reference is None:
        print("Error: Must provide at least one of --optimized or --reference")
        parser.print_help()
        sys.exit(1)

    if args.optimized and not Path(args.optimized).exists():
        print(f"Error: Optimized file not found: {args.optimized}")
        sys.exit(1)

    if args.reference and not Path(args.reference).exists():
        print(f"Error: Reference file not found: {args.reference}")
        sys.exit(1)

    # Import here to defer viser import
    from scripts.vis.grasp_viewer.scene.scene_manager import ViewMode
    from scripts.vis.grasp_viewer.ui.app import GraspViewerApp

    # Create and run app
    app = GraspViewerApp(port=args.port)

    # Load data
    app.load(
        optimized_path=args.optimized,
        reference_path=args.reference,
        asset_dir=args.asset_dir,
    )

    # Set initial view mode
    if args.view_mode == "ghost":
        app.scene_manager.set_view_mode(ViewMode.GHOST)

    # Run
    app.run()


if __name__ == "__main__":
    main()
