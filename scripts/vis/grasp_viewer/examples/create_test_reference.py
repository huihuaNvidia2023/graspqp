#!/usr/bin/env python3
"""Create a test reference trajectory YAML file for development/testing.

This script creates a simple reference trajectory that can be used
to test the grasp viewer without needing real data.

Usage:
    python -m scripts.vis.grasp_viewer.examples.create_test_reference
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from scripts.vis.grasp_viewer.data import save_reference
from scripts.vis.grasp_viewer.data.types import GraspTrajectory, HandStateFormat, HandTrajectory, ObjectTrajectory


def create_test_trajectory(n_frames: int = 50, dt: float = 0.02) -> GraspTrajectory:
    """Create a simple test trajectory.

    Creates a trajectory where:
    - Object is stationary at origin
    - Hand moves in a simple arc toward the object
    """

    # Object: stationary at origin
    object_poses = np.zeros((n_frames, 7), dtype=np.float32)
    object_poses[:, 3] = 1.0  # qw = 1 (identity rotation)

    # Hand: moves in an arc
    t = np.linspace(0, 1, n_frames)

    # Create hand states in graspqp format: [trans(3), rot6d(6), joints(16)]
    # Allegro hand has 16 joints
    n_joints = 16
    hand_states = np.zeros((n_frames, 3 + 6 + n_joints), dtype=np.float32)

    # Translation: arc from (0.2, 0.2, 0.2) to (0.05, 0, 0.1)
    hand_states[:, 0] = 0.2 - 0.15 * t  # x
    hand_states[:, 1] = 0.2 - 0.2 * t  # y
    hand_states[:, 2] = 0.2 - 0.1 * t  # z

    # Rotation: identity (first two columns of identity matrix)
    # rot6d = [1, 0, 0, 0, 1, 0]
    hand_states[:, 3] = 1.0  # col1.x
    hand_states[:, 4] = 0.0  # col1.y
    hand_states[:, 5] = 0.0  # col1.z
    hand_states[:, 6] = 0.0  # col2.x
    hand_states[:, 7] = 1.0  # col2.y
    hand_states[:, 8] = 0.0  # col2.z

    # Joints: gradually close fingers
    # Start open, end in grasp pose
    start_joints = np.array(
        [
            0.0,
            0.2,
            0.2,
            0.2,  # Index
            0.0,
            0.2,
            0.2,
            0.2,  # Middle
            0.0,
            0.2,
            0.2,
            0.2,  # Ring
            1.0,
            0.3,
            0.3,
            0.2,  # Thumb
        ]
    )
    end_joints = np.array(
        [
            0.0,
            0.5,
            0.7,
            0.7,  # Index
            0.0,
            0.5,
            0.7,
            0.7,  # Middle
            0.0,
            0.5,
            0.7,
            0.7,  # Ring
            1.3,
            0.7,
            0.7,
            0.5,  # Thumb
        ]
    )

    for i in range(n_joints):
        hand_states[:, 9 + i] = start_joints[i] + t * (end_joints[i] - start_joints[i])

    # Get paths to allegro assets
    graspqp_assets = Path(__file__).parent.parent.parent.parent.parent / "graspqp" / "assets"

    hand_traj = HandTrajectory(
        name="right_hand",
        hand_type="allegro",
        urdf_path=str(graspqp_assets / "allegro" / "allegro_hand.urdf"),
        mesh_dir=str(graspqp_assets / "allegro" / "meshes"),
        hand_states=hand_states,
        state_format=HandStateFormat.GRASPQP,
    )

    # Look for an object mesh
    objects_dir = Path(__file__).parent.parent.parent.parent.parent / "objects" / "apple"
    obj_mesh = ""
    for pattern in ["*.obj", "*.stl"]:
        matches = list(objects_dir.glob(pattern))
        if matches:
            obj_mesh = str(matches[0])
            break

    obj_traj = ObjectTrajectory(
        mesh_path=obj_mesh,
        scale=1.0,
        poses=object_poses,
    )

    return GraspTrajectory(
        name="test_reference",
        n_frames=n_frames,
        dt=dt,
        hands=[hand_traj],
        obj=obj_traj,
    )


def main():
    """Create and save test reference trajectory."""
    output_dir = Path(__file__).parent / "test_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "test_reference.yaml"

    print("Creating test reference trajectory...")
    trajectory = create_test_trajectory(n_frames=50, dt=0.02)

    print(f"Saving to {output_path}...")
    save_reference(trajectory, str(output_path), embed_arrays=True)

    print("Done!")
    print(f"\nTo test the viewer:")
    print(f"  python -m scripts.vis.grasp_viewer.main --reference {output_path}")


if __name__ == "__main__":
    main()
