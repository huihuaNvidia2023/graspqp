"""Frame label renderer for ghost view mode."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import viser

from scripts.vis.grasp_viewer.renderers.base import BaseRenderer


class LabelRenderer(BaseRenderer):
    """Renders frame labels (t=0, t=1, ...) for ghost view mode."""

    def __init__(self, server: viser.ViserServer, name: str):
        """Initialize label renderer.

        Args:
            server: Viser server instance
            name: Unique name
        """
        super().__init__(server, name)

        # Label handles
        self.label_handles: List[viser.LabelHandle] = []

    def update(
        self, positions: List[np.ndarray], labels: Optional[List[str]] = None, offset: np.ndarray = None
    ) -> None:
        """Update labels at given positions.

        Args:
            positions: List of (3,) positions for each label
            labels: List of label texts. If None, uses "t=0", "t=1", etc.
            offset: (3,) offset to apply to all positions (e.g., [0, 0, 0.1] to place above)
        """
        # Clear existing
        self._clear_labels()

        if not positions:
            return

        if offset is None:
            offset = np.array([0, 0, 0.08])

        if labels is None:
            labels = [f"t={i}" for i in range(len(positions))]

        for i, (pos, text) in enumerate(zip(positions, labels)):
            label_pos = pos + offset
            handle = self.server.scene.add_label(
                name=f"/{self.name}/label_{i}",
                text=text,
                position=label_pos.astype(np.float64),
            )
            handle.visible = self._visible
            self.label_handles.append(handle)

    def set_labels(self, frame_indices: List[int], positions: List[np.ndarray], offset: np.ndarray = None) -> None:
        """Set labels for specific frame indices.

        Args:
            frame_indices: List of frame indices
            positions: List of positions (one per frame)
            offset: Position offset for labels
        """
        labels = [f"t={idx}" for idx in frame_indices]
        self.update(positions, labels, offset)

    def _clear_labels(self) -> None:
        """Clear all label handles."""
        for handle in self.label_handles:
            handle.remove()
        self.label_handles.clear()

    def _update_visibility(self) -> None:
        """Update visibility of all labels."""
        for handle in self.label_handles:
            handle.visible = self._visible

    def remove(self) -> None:
        """Remove all labels."""
        self._clear_labels()


class GhostHandRenderer:
    """Helper class that manages multiple hand renderers for ghost view.

    In ghost view mode, we need to render the same hand at multiple
    frames simultaneously with varying opacity.
    """

    def __init__(self, server: viser.ViserServer, base_name: str, urdf_path: str, mesh_dir: str, n_frames: int):
        """Initialize ghost renderer.

        Args:
            server: Viser server
            base_name: Base name for hands
            urdf_path: Path to URDF
            mesh_dir: Path to meshes
            n_frames: Number of frames to render
        """
        from scripts.vis.grasp_viewer.renderers.hand_renderer import HandRenderer

        self.server = server
        self.base_name = base_name
        self.n_frames = n_frames

        # Create a hand renderer for each frame
        self.hand_renderers: List[HandRenderer] = []
        for i in range(n_frames):
            renderer = HandRenderer(
                server=server,
                name=f"{base_name}_ghost_{i}",
                urdf_path=urdf_path,
                mesh_dir=mesh_dir,
            )
            self.hand_renderers.append(renderer)

        # Label renderer
        self.labels = LabelRenderer(server, f"{base_name}_labels")

    def update_all_frames(
        self,
        hand_trajectory,
        opacity_gradient: bool = True,
        base_opacity: float = 0.2,
        max_opacity: float = 1.0,
        color: Tuple[int, int, int] = (180, 180, 180),
    ) -> None:
        """Update all ghost frames.

        Args:
            hand_trajectory: HandTrajectory object
            opacity_gradient: If True, vary opacity from base to max
            base_opacity: Starting opacity (frame 0)
            max_opacity: Ending opacity (last frame)
            color: RGB color for all frames
        """
        n_frames = min(self.n_frames, hand_trajectory.n_frames)
        label_positions = []

        for i in range(n_frames):
            renderer = self.hand_renderers[i]

            # Calculate opacity
            if opacity_gradient and n_frames > 1:
                t = i / (n_frames - 1)
                opacity = base_opacity + t * (max_opacity - base_opacity)
            else:
                opacity = max_opacity

            renderer.color = color
            renderer.opacity = opacity
            renderer.update(hand_trajectory, frame_idx=i)

            # Get wrist position for label
            # Use the base position from the hand state
            state = hand_trajectory.get_frame(i)
            if hand_trajectory.state_format.name == "GRASPQP":
                wrist_pos = state[:3]
            else:
                # For qpos, we'd need FK - use origin as fallback
                wrist_pos = np.zeros(3)
            label_positions.append(wrist_pos)

        # Update labels
        self.labels.set_labels(list(range(n_frames)), label_positions)

    def set_visible(self, visible: bool) -> None:
        """Set visibility for all ghost frames."""
        for renderer in self.hand_renderers:
            renderer.visible = visible
        self.labels.visible = visible

    def set_labels_visible(self, visible: bool) -> None:
        """Set label visibility."""
        self.labels.visible = visible

    def remove(self) -> None:
        """Remove all ghost renderers."""
        for renderer in self.hand_renderers:
            renderer.remove()
        self.labels.remove()
        self.hand_renderers.clear()
