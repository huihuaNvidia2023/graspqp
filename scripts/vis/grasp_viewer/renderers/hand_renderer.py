"""Hand renderer using URDF and trimesh with viser."""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation as R

from scripts.vis.grasp_viewer.data.types import HandStateFormat, HandTrajectory
from scripts.vis.grasp_viewer.renderers.base import BaseRenderer


class URDFParser:
    """Simple URDF parser to extract link meshes and joint structure."""

    def __init__(self, urdf_path: str, mesh_dir: str):
        """Parse URDF file.

        Args:
            urdf_path: Path to URDF file
            mesh_dir: Directory containing mesh files
        """
        self.urdf_path = Path(urdf_path)
        self.mesh_dir = Path(mesh_dir)

        self.links: Dict[str, Dict] = {}  # link_name -> {mesh, origin, etc.}
        self.joints: Dict[str, Dict] = {}  # joint_name -> {type, parent, child, axis, etc.}
        self.joint_order: List[str] = []  # Ordered list of joint names

        self._parse()

    def _parse(self) -> None:
        """Parse the URDF XML."""
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()

        # Parse links
        for link in root.findall("link"):
            link_name = link.get("name")
            link_data = {"meshes": [], "origins": []}

            # Try visual first, fall back to collision
            visuals = link.findall("visual")
            if not visuals:
                visuals = link.findall("collision")

            for visual in visuals:
                geometry = visual.find("geometry")
                if geometry is not None:
                    mesh_elem = geometry.find("mesh")
                    if mesh_elem is not None:
                        filename = mesh_elem.get("filename", "")
                        scale = mesh_elem.get("scale", "1 1 1")
                        scale = np.array([float(x) for x in scale.split()])

                        # Resolve mesh path
                        mesh_path = self._resolve_mesh_path(filename)
                        if mesh_path:
                            link_data["meshes"].append(
                                {
                                    "path": mesh_path,
                                    "scale": scale,
                                }
                            )

                # Parse origin transform
                origin = visual.find("origin")
                if origin is not None:
                    xyz = origin.get("xyz", "0 0 0")
                    rpy = origin.get("rpy", "0 0 0")
                    xyz = np.array([float(x) for x in xyz.split()])
                    rpy = np.array([float(x) for x in rpy.split()])
                    link_data["origins"].append({"xyz": xyz, "rpy": rpy})
                else:
                    link_data["origins"].append({"xyz": np.zeros(3), "rpy": np.zeros(3)})

            self.links[link_name] = link_data

        # Parse joints
        for joint in root.findall("joint"):
            joint_name = joint.get("name")
            joint_type = joint.get("type")

            parent = joint.find("parent")
            child = joint.find("child")
            axis = joint.find("axis")
            origin = joint.find("origin")
            limit = joint.find("limit")

            joint_data = {
                "type": joint_type,
                "parent": parent.get("link") if parent is not None else None,
                "child": child.get("link") if child is not None else None,
                "axis": np.array([float(x) for x in axis.get("xyz", "0 0 1").split()])
                if axis is not None
                else np.array([0, 0, 1]),
            }

            if origin is not None:
                xyz = origin.get("xyz", "0 0 0")
                rpy = origin.get("rpy", "0 0 0")
                joint_data["xyz"] = np.array([float(x) for x in xyz.split()])
                joint_data["rpy"] = np.array([float(x) for x in rpy.split()])
            else:
                joint_data["xyz"] = np.zeros(3)
                joint_data["rpy"] = np.zeros(3)

            if limit is not None:
                joint_data["lower"] = float(limit.get("lower", -np.pi))
                joint_data["upper"] = float(limit.get("upper", np.pi))

            self.joints[joint_name] = joint_data

            # Track revolute/prismatic joints for FK
            if joint_type in ("revolute", "continuous", "prismatic"):
                self.joint_order.append(joint_name)

    def _resolve_mesh_path(self, filename: str) -> Optional[str]:
        """Resolve mesh filename to full path."""
        # Handle package:// URLs
        if filename.startswith("package://"):
            filename = filename.split("/", 3)[-1]

        # Try various paths
        candidates = [
            self.mesh_dir / filename,
            self.mesh_dir / Path(filename).name,
            self.urdf_path.parent / filename,
            self.urdf_path.parent / Path(filename).name,
        ]

        for path in candidates:
            if path.exists():
                return str(path)

        return None

    @property
    def n_joints(self) -> int:
        """Number of actuated joints."""
        return len(self.joint_order)


class ForwardKinematics:
    """Simple forward kinematics for URDF robot."""

    def __init__(self, urdf: URDFParser):
        """Initialize FK solver.

        Args:
            urdf: Parsed URDF
        """
        self.urdf = urdf
        self._build_kinematic_tree()

    def _build_kinematic_tree(self) -> None:
        """Build kinematic tree structure."""
        # Find root link (link that is never a child)
        children = {j["child"] for j in self.urdf.joints.values() if j["child"]}
        parents = {j["parent"] for j in self.urdf.joints.values() if j["parent"]}
        roots = parents - children
        self.root_link = roots.pop() if roots else list(self.urdf.links.keys())[0]

        # Build parent map
        self.link_parent: Dict[str, Optional[str]] = {self.root_link: None}
        self.link_joint: Dict[str, Optional[str]] = {self.root_link: None}

        for joint_name, joint in self.urdf.joints.items():
            child = joint["child"]
            parent = joint["parent"]
            if child:
                self.link_parent[child] = parent
                self.link_joint[child] = joint_name

    def compute(
        self, joint_angles: np.ndarray, base_pos: np.ndarray = None, base_quat: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """Compute link transforms given joint angles.

        Args:
            joint_angles: (n_joints,) array of joint values
            base_pos: (3,) base position, defaults to origin
            base_quat: (4,) base quaternion (wxyz), defaults to identity

        Returns:
            Dict mapping link names to 4x4 transform matrices
        """
        if base_pos is None:
            base_pos = np.zeros(3)
        if base_quat is None:
            base_quat = np.array([1, 0, 0, 0])  # wxyz

        # Build joint value lookup
        joint_values = {}
        for i, joint_name in enumerate(self.urdf.joint_order):
            if i < len(joint_angles):
                joint_values[joint_name] = joint_angles[i]
            else:
                joint_values[joint_name] = 0.0

        # Base transform
        base_rot = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]]).as_matrix()
        base_transform = np.eye(4)
        base_transform[:3, :3] = base_rot
        base_transform[:3, 3] = base_pos

        # Compute transforms for all links
        transforms = {self.root_link: base_transform}

        # Process links in order (BFS from root)
        visited = {self.root_link}
        queue = [self.root_link]

        while queue:
            link = queue.pop(0)

            # Find child links
            for joint_name, joint in self.urdf.joints.items():
                if joint["parent"] == link and joint["child"] not in visited:
                    child = joint["child"]
                    visited.add(child)
                    queue.append(child)

                    # Compute child transform
                    parent_tf = transforms[link]
                    joint_tf = self._joint_transform(joint, joint_values.get(joint_name, 0.0))
                    transforms[child] = parent_tf @ joint_tf

        return transforms

    def _joint_transform(self, joint: Dict, value: float) -> np.ndarray:
        """Compute transform for a joint at given value."""
        # Origin transform
        xyz = joint.get("xyz", np.zeros(3))
        rpy = joint.get("rpy", np.zeros(3))

        origin_rot = R.from_euler("xyz", rpy).as_matrix()
        origin_tf = np.eye(4)
        origin_tf[:3, :3] = origin_rot
        origin_tf[:3, 3] = xyz

        # Joint motion transform
        motion_tf = np.eye(4)
        joint_type = joint.get("type", "fixed")
        axis = joint.get("axis", np.array([0, 0, 1]))

        if joint_type in ("revolute", "continuous"):
            motion_rot = R.from_rotvec(value * axis).as_matrix()
            motion_tf[:3, :3] = motion_rot
        elif joint_type == "prismatic":
            motion_tf[:3, 3] = value * axis

        return origin_tf @ motion_tf


class HandRenderer(BaseRenderer):
    """Renders a hand model using URDF and viser meshes with transparency support."""

    def __init__(self, server: viser.ViserServer, name: str, urdf_path: str, mesh_dir: str):
        """Initialize hand renderer.

        Args:
            server: Viser server instance
            name: Unique name for this hand (e.g., "right_hand", "ref_hand")
            urdf_path: Path to URDF file
            mesh_dir: Path to mesh directory
        """
        super().__init__(server, name)

        self.urdf_path = urdf_path
        self.mesh_dir = mesh_dir

        # Parse URDF
        self.urdf = URDFParser(urdf_path, mesh_dir)
        self.fk = ForwardKinematics(self.urdf)

        # Load meshes
        self.link_meshes: Dict[str, List[trimesh.Trimesh]] = {}
        self._load_meshes()

        # Viser mesh handles
        self.mesh_handles: Dict[str, List[viser.MeshHandle]] = {}
        self._initialized = False

    def _load_meshes(self) -> None:
        """Load mesh files for each link."""
        for link_name, link_data in self.urdf.links.items():
            meshes = []
            for mesh_info in link_data["meshes"]:
                try:
                    mesh = trimesh.load(mesh_info["path"], process=False)
                    if isinstance(mesh, trimesh.Scene):
                        # Combine scene into single mesh
                        mesh = mesh.dump(concatenate=True)
                    mesh.apply_scale(mesh_info["scale"])
                    meshes.append(mesh)
                except Exception as e:
                    print(f"Warning: Could not load mesh {mesh_info['path']}: {e}")

            if meshes:
                self.link_meshes[link_name] = meshes

    def _create_mesh_handles(self) -> None:
        """Create viser mesh handles for all links."""
        for link_name, meshes in self.link_meshes.items():
            link_handles = []
            for i, mesh in enumerate(meshes):
                handle = self.server.scene.add_mesh_simple(
                    name=f"/{self.name}/{link_name}/{i}",
                    vertices=mesh.vertices.astype(np.float32),
                    faces=mesh.faces.astype(np.uint32),
                    color=self._color,
                    opacity=self._opacity,
                )
                link_handles.append(handle)
            self.mesh_handles[link_name] = link_handles

        self._initialized = True

    def update(
        self,
        hand_trajectory: HandTrajectory,
        frame_idx: int = 0,
        base_pos: Optional[np.ndarray] = None,
        base_quat: Optional[np.ndarray] = None,
    ) -> None:
        """Update hand pose from trajectory.

        Args:
            hand_trajectory: HandTrajectory object
            frame_idx: Frame index to display
            base_pos: Optional override for base position
            base_quat: Optional override for base quaternion (wxyz)
        """
        # Initialize mesh handles if not done
        if not self._initialized:
            self._create_mesh_handles()

        # Get state for this frame
        state = hand_trajectory.get_frame(frame_idx)

        # Parse state based on format
        if hand_trajectory.state_format == HandStateFormat.QPOS:
            joint_angles = state
            if base_pos is None:
                base_pos = np.zeros(3)
            if base_quat is None:
                base_quat = np.array([1, 0, 0, 0])
        else:  # GRASPQP format
            trans = state[:3]
            rot6d = state[3:9]
            joint_angles = state[9:]

            # Convert rot6d to rotation matrix
            col1 = rot6d[:3] / (np.linalg.norm(rot6d[:3]) + 1e-8)
            col2_raw = rot6d[3:6]
            col2 = col2_raw - np.dot(col2_raw, col1) * col1
            col2 = col2 / (np.linalg.norm(col2) + 1e-8)
            col3 = np.cross(col1, col2)
            rot_mat = np.column_stack([col1, col2, col3])

            # Convert to quaternion (wxyz)
            quat_xyzw = R.from_matrix(rot_mat).as_quat()
            base_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            base_pos = trans

        # Apply base transform from trajectory if available
        if hand_trajectory.base_transform is not None and base_pos is None:
            bt = hand_trajectory.base_transform
            base_pos = bt[:3]
            base_quat = bt[3:7]

        # Compute FK
        transforms = self.fk.compute(joint_angles, base_pos, base_quat)

        # Update mesh positions
        for link_name, link_handles in self.mesh_handles.items():
            if link_name not in transforms:
                continue

            tf = transforms[link_name]
            pos = tf[:3, 3]
            rot_mat = tf[:3, :3]
            quat_xyzw = R.from_matrix(rot_mat).as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

            # Apply visual origin transforms
            link_data = self.urdf.links.get(link_name, {})
            origins = link_data.get("origins", [])

            for i, handle in enumerate(link_handles):
                # Get origin for this visual
                if i < len(origins):
                    origin = origins[i]
                    origin_xyz = origin.get("xyz", np.zeros(3))
                    origin_rpy = origin.get("rpy", np.zeros(3))
                    origin_rot = R.from_euler("xyz", origin_rpy).as_matrix()

                    # Combined transform
                    origin_tf = np.eye(4)
                    origin_tf[:3, :3] = origin_rot
                    origin_tf[:3, 3] = origin_xyz

                    combined = tf @ origin_tf
                    pos = combined[:3, 3]
                    rot_mat = combined[:3, :3]
                    quat_xyzw = R.from_matrix(rot_mat).as_quat()
                    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

                handle.position = pos
                handle.wxyz = quat_wxyz

    def update_from_qpos(
        self, qpos: np.ndarray, base_pos: Optional[np.ndarray] = None, base_quat: Optional[np.ndarray] = None
    ) -> None:
        """Update hand pose directly from joint angles.

        Args:
            qpos: Joint angles
            base_pos: Base position
            base_quat: Base quaternion (wxyz)
        """
        if not self._initialized:
            self._create_mesh_handles()

        transforms = self.fk.compute(qpos, base_pos, base_quat)

        for link_name, link_handles in self.mesh_handles.items():
            if link_name not in transforms:
                continue

            tf = transforms[link_name]
            pos = tf[:3, 3]
            rot_mat = tf[:3, :3]
            quat_xyzw = R.from_matrix(rot_mat).as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

            link_data = self.urdf.links.get(link_name, {})
            origins = link_data.get("origins", [])

            for i, handle in enumerate(link_handles):
                if i < len(origins):
                    origin = origins[i]
                    origin_xyz = origin.get("xyz", np.zeros(3))
                    origin_rpy = origin.get("rpy", np.zeros(3))
                    origin_rot = R.from_euler("xyz", origin_rpy).as_matrix()

                    origin_tf = np.eye(4)
                    origin_tf[:3, :3] = origin_rot
                    origin_tf[:3, 3] = origin_xyz

                    combined = tf @ origin_tf
                    pos = combined[:3, 3]
                    rot_mat = combined[:3, :3]
                    quat_xyzw = R.from_matrix(rot_mat).as_quat()
                    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

                handle.position = pos
                handle.wxyz = quat_wxyz

    def _update_visibility(self) -> None:
        """Update visibility of all mesh handles."""
        for handles in self.mesh_handles.values():
            for handle in handles:
                handle.visible = self._visible

    def _update_appearance(self) -> None:
        """Update color and opacity of all mesh handles."""
        for handles in self.mesh_handles.values():
            for handle in handles:
                # viser mesh handles use separate color and opacity
                handle.color = self._color
                handle.opacity = self._opacity

    def remove(self) -> None:
        """Remove all mesh handles."""
        for handles in self.mesh_handles.values():
            for handle in handles:
                handle.remove()
        self.mesh_handles.clear()
        self._initialized = False

    @property
    def n_joints(self) -> int:
        """Number of actuated joints."""
        return self.urdf.n_joints
