"""Object mesh renderer using viser."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation as R

from scripts.vis.grasp_viewer.data.types import ObjectTrajectory
from scripts.vis.grasp_viewer.renderers.base import BaseRenderer


class ObjectRenderer(BaseRenderer):
    """Renders an object mesh with viser."""

    def __init__(self, server: viser.ViserServer, name: str, mesh_path: str, scale: float = 1.0):
        """Initialize object renderer.

        Args:
            server: Viser server instance
            name: Unique name for this object
            mesh_path: Path to mesh file (.obj, .stl, etc.)
            scale: Scale factor for the mesh
        """
        super().__init__(server, name)

        self.mesh_path = mesh_path
        self.scale = scale

        # Load mesh
        self.mesh: Optional[trimesh.Trimesh] = None
        self._load_mesh()

        # Viser handle
        self.mesh_handle: Optional[viser.MeshHandle] = None
        self._initialized = False

        # Default object color (greenish)
        self._color = (128, 200, 128)

    def _load_mesh(self) -> None:
        """Load the mesh file."""
        if not self.mesh_path or not Path(self.mesh_path).exists():
            print(f"Warning: Object mesh not found: {self.mesh_path}")
            return

        try:
            mesh = trimesh.load(self.mesh_path, process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)

            # Apply scale
            if self.scale != 1.0:
                mesh.apply_scale(self.scale)

            self.mesh = mesh
        except Exception as e:
            print(f"Warning: Could not load mesh {self.mesh_path}: {e}")

    def _create_mesh_handle(self) -> None:
        """Create viser mesh handle."""
        if self.mesh is None:
            return

        self.mesh_handle = self.server.scene.add_mesh_simple(
            name=f"/{self.name}/mesh",
            vertices=self.mesh.vertices.astype(np.float32),
            faces=self.mesh.faces.astype(np.uint32),
            color=self._color,
            opacity=self._opacity,
        )
        self._initialized = True

    def update(
        self,
        obj_trajectory: Optional[ObjectTrajectory] = None,
        frame_idx: int = 0,
        position: Optional[np.ndarray] = None,
        quaternion: Optional[np.ndarray] = None,
    ) -> None:
        """Update object pose.

        Args:
            obj_trajectory: ObjectTrajectory to read pose from
            frame_idx: Frame index
            position: Override position (3,)
            quaternion: Override quaternion wxyz (4,)
        """
        if not self._initialized:
            self._create_mesh_handle()

        if self.mesh_handle is None:
            return

        # Get pose from trajectory or use overrides
        if obj_trajectory is not None and obj_trajectory.poses is not None:
            pose = obj_trajectory.get_frame(frame_idx)
            pos = pose[:3]
            quat_wxyz = pose[3:7]
        else:
            pos = position if position is not None else np.zeros(3)
            quat_wxyz = quaternion if quaternion is not None else np.array([1, 0, 0, 0])

        self.mesh_handle.position = pos.astype(np.float64)
        self.mesh_handle.wxyz = quat_wxyz.astype(np.float64)

    def update_from_pose(self, position: np.ndarray, quaternion: np.ndarray) -> None:
        """Update object pose directly.

        Args:
            position: (3,) position
            quaternion: (4,) quaternion wxyz
        """
        if not self._initialized:
            self._create_mesh_handle()

        if self.mesh_handle is None:
            return

        self.mesh_handle.position = position.astype(np.float64)
        self.mesh_handle.wxyz = quaternion.astype(np.float64)

    def _update_visibility(self) -> None:
        """Update visibility."""
        if self.mesh_handle is not None:
            self.mesh_handle.visible = self._visible

    def _update_appearance(self) -> None:
        """Update color and opacity."""
        if self.mesh_handle is not None:
            self.mesh_handle.color = self._color
            self.mesh_handle.opacity = self._opacity

    def remove(self) -> None:
        """Remove mesh handle."""
        if self.mesh_handle is not None:
            self.mesh_handle.remove()
            self.mesh_handle = None
        self._initialized = False
