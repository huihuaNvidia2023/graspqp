"""Main application for grasp trajectory visualization."""

import time
from typing import Optional

import viser

from scripts.vis.grasp_viewer.data import TrajectoryBundle, load_trajectories
from scripts.vis.grasp_viewer.scene import SceneManager
from scripts.vis.grasp_viewer.scene.scene_manager import ViewMode
from scripts.vis.grasp_viewer.ui.control_panel import ControlPanel


class GraspViewerApp:
    """Main application class for grasp trajectory visualization."""

    def __init__(self, port: int = 8080):
        """Initialize the application.

        Args:
            port: Port for viser server
        """
        self.port = port

        # Create viser server
        self.server = viser.ViserServer(port=port)
        print(f"Viser server starting at http://localhost:{port}")

        # Create components
        self.scene_manager = SceneManager(self.server)
        self.control_panel = ControlPanel(self.server)

        # Data
        self.bundle: Optional[TrajectoryBundle] = None

        # Connect callbacks
        self._setup_callbacks()

    def _setup_callbacks(self) -> None:
        """Setup UI callbacks."""
        # Trajectory selection
        self.control_panel.set_on_trajectory_change(self._on_trajectory_change)

        # View mode
        self.control_panel.set_on_view_mode_change(self._on_view_mode_change)

        # Reference overlay
        self.control_panel.set_on_reference_toggle(self._on_reference_toggle)
        self.control_panel.set_on_reference_opacity(self._on_reference_opacity)

        # Visibility
        self.control_panel.set_on_visibility_change(self._on_visibility_change)

        # Playback frame changes
        self.control_panel.playback.set_on_frame_change(self._on_frame_change)

    def load(
        self,
        optimized_path: Optional[str] = None,
        reference_path: Optional[str] = None,
        asset_dir: Optional[str] = None,
    ) -> None:
        """Load trajectory data.

        Args:
            optimized_path: Path to .dexgrasp.pt file
            reference_path: Path to reference YAML file
            asset_dir: Path to graspqp assets directory
        """
        import sys

        print("Loading trajectories...", flush=True)

        if optimized_path:
            print(f"  Optimized: {optimized_path}", flush=True)
        if reference_path:
            print(f"  Reference: {reference_path}", flush=True)

        self.bundle = load_trajectories(
            optimized_path=optimized_path,
            reference_path=reference_path,
            asset_dir=asset_dir,
        )

        # Update UI
        if self.bundle.has_optimized:
            options = self.bundle.get_trajectory_names()
            energies = [t.energy for t in self.bundle.optimized]
            self.control_panel.set_trajectory_options(options, energies)

            # Set playback info from selected trajectory
            selected = self.bundle.selected
            if selected:
                self.control_panel.set_playback_info(selected.n_frames, selected.dt)

        # Load into scene
        self.scene_manager.load_bundle(self.bundle)

        print(f"Loaded {self.bundle.n_optimized} optimized trajectories")
        if self.bundle.has_reference:
            print(f"Loaded reference trajectory with {self.bundle.reference.n_frames} frames")

    def run(self) -> None:
        """Run the main application loop."""
        print(f"\nGrasp Viewer running at http://localhost:{self.port}")
        print("Press Ctrl+C to exit\n")

        try:
            while True:
                # Update playback
                self.control_panel.update()

                # Small sleep to prevent busy waiting
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nShutting down...")

    # === Callback Handlers ===

    def _on_trajectory_change(self, index: int) -> None:
        """Handle trajectory selection change."""
        if self.bundle is None:
            return

        if self.scene_manager.select_trajectory(index):
            # Update playback info
            selected = self.bundle.selected
            if selected:
                self.control_panel.set_playback_info(selected.n_frames, selected.dt)

                # Update energy display
                if selected.energy is not None:
                    self.control_panel._energy_text.value = f"{selected.energy:.4f}"

    def _on_view_mode_change(self, mode: ViewMode) -> None:
        """Handle view mode change."""
        self.scene_manager.set_view_mode(mode)

    def _on_reference_toggle(self, show: bool) -> None:
        """Handle reference visibility toggle."""
        self.scene_manager.set_show_reference(show)

    def _on_reference_opacity(self, opacity: float) -> None:
        """Handle reference opacity change."""
        self.scene_manager.set_reference_opacity(opacity)

    def _on_visibility_change(self, item: str, visible: bool) -> None:
        """Handle visibility toggle."""
        if item == "object":
            self.scene_manager.set_object_visible(visible)
        elif item == "labels":
            self.scene_manager.set_labels_visible(visible)
        elif item == "contacts":
            # TODO: Implement contact visibility
            pass

    def _on_frame_change(self, frame: int) -> None:
        """Handle playback frame change."""
        self.scene_manager.update_frame(frame)
