"""Scene manager that orchestrates all renderers."""

from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
import viser

from scripts.vis.grasp_viewer.data.types import GraspTrajectory, HandTrajectory, ObjectTrajectory, TrajectoryBundle
from scripts.vis.grasp_viewer.renderers import ContactRenderer, HandRenderer, LabelRenderer, ObjectRenderer
from scripts.vis.grasp_viewer.renderers.label_renderer import GhostHandRenderer


class ViewMode(Enum):
    """View mode for trajectory visualization."""

    PLAYBACK = auto()  # Single frame at a time
    GHOST = auto()  # All frames visible with opacity gradient


class SceneManager:
    """Manages the 3D scene for grasp trajectory visualization."""

    # Color schemes
    OPTIMIZED_COLOR = (180, 180, 180)  # Gray for optimized
    REFERENCE_COLOR = (100, 150, 200)  # Blue-ish for reference
    OBJECT_COLOR = (128, 200, 128)  # Green for object

    def __init__(self, server: viser.ViserServer):
        """Initialize scene manager.

        Args:
            server: Viser server instance
        """
        self.server = server

        # Current data
        self.bundle: Optional[TrajectoryBundle] = None

        # View state
        self.view_mode = ViewMode.PLAYBACK
        self.current_frame = 0
        self.show_reference = True
        self.reference_opacity = 0.3

        # Renderers for optimized trajectory
        self.opt_hand_renderers: List[HandRenderer] = []
        self.opt_object_renderer: Optional[ObjectRenderer] = None
        self.opt_ghost_renderers: List[GhostHandRenderer] = []
        self.opt_labels: Optional[LabelRenderer] = None

        # Renderers for reference trajectory
        self.ref_hand_renderers: List[HandRenderer] = []
        self.ref_object_renderer: Optional[ObjectRenderer] = None
        self.ref_ghost_renderers: List[GhostHandRenderer] = []
        self.ref_labels: Optional[LabelRenderer] = None

        # Contact renderer (for optimized only)
        self.contact_renderer: Optional[ContactRenderer] = None

        # Add coordinate frame at origin
        self._add_world_frame()

    def _add_world_frame(self) -> None:
        """Add a coordinate frame at the world origin."""
        self.server.scene.add_frame(
            "/world_frame",
            axes_length=0.1,
            axes_radius=0.003,
        )

    def load_bundle(self, bundle: TrajectoryBundle) -> None:
        """Load a trajectory bundle for visualization.

        Args:
            bundle: TrajectoryBundle with reference and/or optimized trajectories
        """
        # Clear existing renderers
        self._clear_all()

        self.bundle = bundle

        # Setup optimized trajectory renderers
        if bundle.has_optimized:
            self._setup_optimized_renderers(bundle.selected)

        # Setup reference trajectory renderers
        if bundle.has_reference:
            self._setup_reference_renderers(bundle.reference)

        # Initial update
        self.update_frame(0)

    def _setup_optimized_renderers(self, trajectory: GraspTrajectory) -> None:
        """Setup renderers for optimized trajectory."""
        # Hand renderers
        for i, hand in enumerate(trajectory.hands):
            if hand.urdf_path:
                renderer = HandRenderer(
                    self.server,
                    name=f"opt_hand_{i}",
                    urdf_path=hand.urdf_path,
                    mesh_dir=hand.mesh_dir,
                )
                renderer.color = self.OPTIMIZED_COLOR
                self.opt_hand_renderers.append(renderer)

                # Ghost renderer for ghost mode
                if trajectory.n_frames > 1:
                    ghost = GhostHandRenderer(
                        self.server,
                        base_name=f"opt_ghost_{i}",
                        urdf_path=hand.urdf_path,
                        mesh_dir=hand.mesh_dir,
                        n_frames=trajectory.n_frames,
                    )
                    self.opt_ghost_renderers.append(ghost)

        # Object renderer
        if trajectory.obj.mesh_path:
            self.opt_object_renderer = ObjectRenderer(
                self.server,
                name="opt_object",
                mesh_path=trajectory.obj.mesh_path,
                scale=trajectory.obj.scale,
            )
            self.opt_object_renderer.color = self.OBJECT_COLOR

        # Labels for ghost mode
        self.opt_labels = LabelRenderer(self.server, "opt_labels")

        # Contact renderer
        self.contact_renderer = ContactRenderer(self.server, "contacts")

    def _setup_reference_renderers(self, trajectory: GraspTrajectory) -> None:
        """Setup renderers for reference trajectory."""
        # Hand renderers
        for i, hand in enumerate(trajectory.hands):
            if hand.urdf_path:
                renderer = HandRenderer(
                    self.server,
                    name=f"ref_hand_{i}",
                    urdf_path=hand.urdf_path,
                    mesh_dir=hand.mesh_dir,
                )
                renderer.color = self.REFERENCE_COLOR
                renderer.opacity = self.reference_opacity
                self.ref_hand_renderers.append(renderer)

                # Ghost renderer
                if trajectory.n_frames > 1:
                    ghost = GhostHandRenderer(
                        self.server,
                        base_name=f"ref_ghost_{i}",
                        urdf_path=hand.urdf_path,
                        mesh_dir=hand.mesh_dir,
                        n_frames=trajectory.n_frames,
                    )
                    self.ref_ghost_renderers.append(ghost)

        # Object renderer (reference uses same object typically)
        if trajectory.obj.mesh_path and self.opt_object_renderer is None:
            self.ref_object_renderer = ObjectRenderer(
                self.server,
                name="ref_object",
                mesh_path=trajectory.obj.mesh_path,
                scale=trajectory.obj.scale,
            )
            self.ref_object_renderer.color = self.OBJECT_COLOR
            self.ref_object_renderer.opacity = self.reference_opacity

        # Labels
        self.ref_labels = LabelRenderer(self.server, "ref_labels")

    def _clear_all(self) -> None:
        """Clear all renderers."""
        for r in self.opt_hand_renderers:
            r.remove()
        self.opt_hand_renderers.clear()

        for g in self.opt_ghost_renderers:
            g.remove()
        self.opt_ghost_renderers.clear()

        if self.opt_object_renderer:
            self.opt_object_renderer.remove()
            self.opt_object_renderer = None

        if self.opt_labels:
            self.opt_labels.remove()
            self.opt_labels = None

        for r in self.ref_hand_renderers:
            r.remove()
        self.ref_hand_renderers.clear()

        for g in self.ref_ghost_renderers:
            g.remove()
        self.ref_ghost_renderers.clear()

        if self.ref_object_renderer:
            self.ref_object_renderer.remove()
            self.ref_object_renderer = None

        if self.ref_labels:
            self.ref_labels.remove()
            self.ref_labels = None

        if self.contact_renderer:
            self.contact_renderer.remove()
            self.contact_renderer = None

    def select_trajectory(self, index: int) -> bool:
        """Select an optimized trajectory by index.

        Args:
            index: Trajectory index

        Returns:
            True if selection changed
        """
        if self.bundle is None or not self.bundle.select(index):
            return False

        # Re-setup optimized renderers with new trajectory
        for r in self.opt_hand_renderers:
            r.remove()
        self.opt_hand_renderers.clear()

        for g in self.opt_ghost_renderers:
            g.remove()
        self.opt_ghost_renderers.clear()

        if self.opt_object_renderer:
            self.opt_object_renderer.remove()
            self.opt_object_renderer = None

        self._setup_optimized_renderers(self.bundle.selected)
        self.update_frame(self.current_frame)

        return True

    def set_view_mode(self, mode: ViewMode) -> None:
        """Set the view mode.

        Args:
            mode: ViewMode.PLAYBACK or ViewMode.GHOST
        """
        if mode == self.view_mode:
            return

        self.view_mode = mode
        self._update_renderers_for_mode()

    def _update_renderers_for_mode(self) -> None:
        """Update renderer visibility based on view mode."""
        if self.view_mode == ViewMode.PLAYBACK:
            # Show single-frame renderers, hide ghost renderers
            for r in self.opt_hand_renderers:
                r.visible = True
            for g in self.opt_ghost_renderers:
                g.set_visible(False)
            if self.opt_labels:
                self.opt_labels.visible = False

            for r in self.ref_hand_renderers:
                r.visible = self.show_reference
            for g in self.ref_ghost_renderers:
                g.set_visible(False)
            if self.ref_labels:
                self.ref_labels.visible = False

        else:  # GHOST mode
            # Hide single-frame renderers, show ghost renderers
            for r in self.opt_hand_renderers:
                r.visible = False
            for g in self.opt_ghost_renderers:
                g.set_visible(True)
            if self.opt_labels:
                self.opt_labels.visible = True

            for r in self.ref_hand_renderers:
                r.visible = False
            for g in self.ref_ghost_renderers:
                g.set_visible(self.show_reference)
            if self.ref_labels:
                self.ref_labels.visible = self.show_reference

            # Update ghost renderers with all frames
            self._update_ghost_mode()

    def _update_ghost_mode(self) -> None:
        """Update ghost mode renderers to show all frames."""
        if self.bundle is None:
            return

        # Update optimized ghost renderers
        if self.bundle.selected:
            traj = self.bundle.selected
            for i, ghost in enumerate(self.opt_ghost_renderers):
                if i < len(traj.hands):
                    ghost.update_all_frames(
                        traj.hands[i],
                        opacity_gradient=True,
                        base_opacity=0.2,
                        max_opacity=1.0,
                        color=self.OPTIMIZED_COLOR,
                    )

        # Update reference ghost renderers
        if self.bundle.reference and self.show_reference:
            traj = self.bundle.reference
            for i, ghost in enumerate(self.ref_ghost_renderers):
                if i < len(traj.hands):
                    ghost.update_all_frames(
                        traj.hands[i],
                        opacity_gradient=False,  # Uniform opacity for reference
                        base_opacity=self.reference_opacity,
                        max_opacity=self.reference_opacity,
                        color=self.REFERENCE_COLOR,
                    )

    def update_frame(self, frame_idx: int) -> None:
        """Update visualization to show a specific frame.

        Args:
            frame_idx: Frame index
        """
        self.current_frame = frame_idx

        if self.bundle is None:
            return

        # In ghost mode, we don't update per-frame (all frames are shown)
        if self.view_mode == ViewMode.GHOST:
            return

        # Update optimized trajectory
        if self.bundle.selected:
            traj = self.bundle.selected
            frame_idx = min(frame_idx, traj.n_frames - 1)

            for i, renderer in enumerate(self.opt_hand_renderers):
                if i < len(traj.hands):
                    renderer.update(traj.hands[i], frame_idx)

            if self.opt_object_renderer:
                self.opt_object_renderer.update(traj.obj, frame_idx)

        # Update reference trajectory
        if self.bundle.reference and self.show_reference:
            traj = self.bundle.reference
            ref_frame = min(frame_idx, traj.n_frames - 1)

            for i, renderer in enumerate(self.ref_hand_renderers):
                if i < len(traj.hands):
                    renderer.update(traj.hands[i], ref_frame)

            if self.ref_object_renderer:
                self.ref_object_renderer.update(traj.obj, ref_frame)

    def set_show_reference(self, show: bool) -> None:
        """Set whether to show reference trajectory.

        Args:
            show: Whether to show reference
        """
        self.show_reference = show

        for r in self.ref_hand_renderers:
            r.visible = show and (self.view_mode == ViewMode.PLAYBACK)

        for g in self.ref_ghost_renderers:
            g.set_visible(show and (self.view_mode == ViewMode.GHOST))

        if self.ref_object_renderer:
            self.ref_object_renderer.visible = show

        if self.ref_labels:
            self.ref_labels.visible = show and (self.view_mode == ViewMode.GHOST)

    def set_reference_opacity(self, opacity: float) -> None:
        """Set reference trajectory opacity.

        Args:
            opacity: Opacity value (0.0 - 1.0)
        """
        self.reference_opacity = opacity

        for r in self.ref_hand_renderers:
            r.opacity = opacity

        if self.ref_object_renderer:
            self.ref_object_renderer.opacity = opacity

        # Update ghost renderers if in ghost mode
        if self.view_mode == ViewMode.GHOST:
            self._update_ghost_mode()

    def set_object_visible(self, visible: bool) -> None:
        """Set object visibility."""
        if self.opt_object_renderer:
            self.opt_object_renderer.visible = visible
        if self.ref_object_renderer:
            self.ref_object_renderer.visible = visible and self.show_reference

    def set_labels_visible(self, visible: bool) -> None:
        """Set frame labels visibility (ghost mode only)."""
        if self.opt_labels:
            self.opt_labels.visible = visible and (self.view_mode == ViewMode.GHOST)
        if self.ref_labels:
            self.ref_labels.visible = visible and self.show_reference and (self.view_mode == ViewMode.GHOST)

        for g in self.opt_ghost_renderers:
            g.set_labels_visible(visible)
        for g in self.ref_ghost_renderers:
            g.set_labels_visible(visible and self.show_reference)

    @property
    def n_frames(self) -> int:
        """Number of frames in current trajectory."""
        if self.bundle and self.bundle.selected:
            return self.bundle.selected.n_frames
        return 1

    @property
    def dt(self) -> float:
        """Time step of current trajectory."""
        if self.bundle and self.bundle.selected:
            return self.bundle.selected.dt
        return 0.01
