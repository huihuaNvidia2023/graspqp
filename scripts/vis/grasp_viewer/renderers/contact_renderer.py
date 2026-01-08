"""Contact points and normals renderer."""

from typing import List, Optional, Tuple

import numpy as np
import viser

from scripts.vis.grasp_viewer.renderers.base import BaseRenderer


class ContactRenderer(BaseRenderer):
    """Renders contact points and optionally contact normals."""

    def __init__(self, server: viser.ViserServer, name: str, point_size: float = 0.008, normal_length: float = 0.03):
        """Initialize contact renderer.

        Args:
            server: Viser server instance
            name: Unique name
            point_size: Radius of contact point spheres
            normal_length: Length of normal arrows
        """
        super().__init__(server, name)

        self.point_size = point_size
        self.normal_length = normal_length

        # Contact data
        self.points: Optional[np.ndarray] = None  # (N, 3)
        self.normals: Optional[np.ndarray] = None  # (N, 3)

        # Viser handles
        self.point_handles: List[viser.SceneNodeHandle] = []
        self.normal_handles: List[viser.SceneNodeHandle] = []

        # Show normals flag
        self._show_normals = False

        # Contact point color (orange)
        self._color = (255, 140, 0)
        # Normal color (blue)
        self._normal_color = (0, 100, 255)

    @property
    def show_normals(self) -> bool:
        """Whether to show contact normals."""
        return self._show_normals

    @show_normals.setter
    def show_normals(self, value: bool) -> None:
        """Set normal visibility."""
        self._show_normals = value
        for handle in self.normal_handles:
            handle.visible = value and self._visible

    def update(self, points: np.ndarray, normals: Optional[np.ndarray] = None) -> None:
        """Update contact points and normals.

        Args:
            points: (N, 3) contact point positions
            normals: (N, 3) contact normal directions (optional)
        """
        # Clear existing handles
        self._clear_handles()

        self.points = points
        self.normals = normals

        if points is None or len(points) == 0:
            return

        # Create point spheres
        for i, pt in enumerate(points):
            handle = self.server.scene.add_icosphere(
                name=f"/{self.name}/point_{i}",
                radius=self.point_size,
                color=self._color,
                position=pt.astype(np.float64),
            )
            self.point_handles.append(handle)

        # Create normal arrows if provided
        if normals is not None and len(normals) == len(points):
            for i, (pt, normal) in enumerate(zip(points, normals)):
                # Normalize direction
                normal = normal / (np.linalg.norm(normal) + 1e-8)
                end_pt = pt + normal * self.normal_length

                # Use spline for line visualization
                handle = self.server.scene.add_spline_catmull_rom(
                    name=f"/{self.name}/normal_{i}",
                    positions=np.array([pt, end_pt], dtype=np.float64),
                    color=self._normal_color,
                    line_width=2.0,
                )
                handle.visible = self._show_normals and self._visible
                self.normal_handles.append(handle)

    def _clear_handles(self) -> None:
        """Clear all handles."""
        for handle in self.point_handles:
            handle.remove()
        for handle in self.normal_handles:
            handle.remove()
        self.point_handles.clear()
        self.normal_handles.clear()

    def _update_visibility(self) -> None:
        """Update visibility."""
        for handle in self.point_handles:
            handle.visible = self._visible
        for handle in self.normal_handles:
            handle.visible = self._show_normals and self._visible

    def _update_appearance(self) -> None:
        """Update appearance (limited for spheres)."""
        # Note: viser icospheres don't support dynamic color changes easily
        # Would need to recreate handles for color changes
        pass

    def remove(self) -> None:
        """Remove all handles."""
        self._clear_handles()
        self.points = None
        self.normals = None
