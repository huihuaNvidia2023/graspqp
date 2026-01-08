#!/usr/bin/env python3
"""Standalone launcher for grasp trajectory visualizer.

Usage:
    python run_grasp_viewer.py --optimized path/to/optimized.dexgrasp.pt
    python run_grasp_viewer.py --optimized data.pt --reference ref.yaml
"""

import sys
from pathlib import Path

# Fix namespace collision with ROS scripts package
# Remove ROS paths temporarily to allow local scripts to be found first
_ros_paths = [p for p in sys.path if "/opt/ros" in p or "dist-packages" in p]
for p in _ros_paths:
    sys.path.remove(p)

# Add local repo root to front of path
_REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT))

# Now import and run main
from scripts.vis.grasp_viewer.main import main

if __name__ == "__main__":
    main()
