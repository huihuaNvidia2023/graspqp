"""Control panel UI using viser GUI."""

from typing import Callable, List, Optional

import viser

from scripts.vis.grasp_viewer.scene.scene_manager import ViewMode
from scripts.vis.grasp_viewer.ui.playback import PlaybackController


class ControlPanel:
    """GUI control panel for the grasp viewer."""

    def __init__(self, server: viser.ViserServer):
        """Initialize control panel.

        Args:
            server: Viser server instance
        """
        self.server = server

        # Create playback controller
        self.playback = PlaybackController()

        # GUI handles
        self._trajectory_dropdown: Optional[viser.GuiDropdownHandle] = None
        self._energy_text: Optional[viser.GuiInputHandle] = None

        # Callbacks
        self._on_trajectory_change: Optional[Callable[[int], None]] = None
        self._on_view_mode_change: Optional[Callable[[ViewMode], None]] = None
        self._on_reference_toggle: Optional[Callable[[bool], None]] = None
        self._on_reference_opacity: Optional[Callable[[float], None]] = None
        self._on_visibility_change: Optional[Callable[[str, bool], None]] = None

        # Build UI
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the control panel UI."""

        # === Trajectory Selection ===
        with self.server.gui.add_folder("Trajectory Selection"):
            self._trajectory_dropdown = self.server.gui.add_dropdown(
                "Trajectory",
                options=["No trajectories loaded"],
                initial_value="No trajectories loaded",
            )
            self._trajectory_dropdown.on_update(self._handle_trajectory_change)

            self._energy_text = self.server.gui.add_text(
                "Energy",
                initial_value="N/A",
                disabled=True,
            )

        # === View Mode ===
        with self.server.gui.add_folder("View Mode"):
            self._view_mode_buttons = self.server.gui.add_button_group(
                "Mode",
                options=["Playback", "Ghost"],
            )
            self._view_mode_buttons.on_click(self._handle_view_mode_change)

        # === Playback Controls ===
        with self.server.gui.add_folder("Playback Controls"):
            self._frame_slider = self.server.gui.add_slider(
                "Frame",
                min=0,
                max=1,
                step=1,
                initial_value=0,
            )
            self._frame_slider.on_update(self._handle_frame_slider)

            self._frame_text = self.server.gui.add_text(
                "Time",
                initial_value="0.00s / 0.00s",
                disabled=True,
            )

            # Playback buttons
            with self.server.gui.add_folder("Transport", expand_by_default=True):
                self._play_button = self.server.gui.add_button("▶ Play")
                self._play_button.on_click(self._handle_play_click)

                self._stop_button = self.server.gui.add_button("⏹ Stop")
                self._stop_button.on_click(self._handle_stop_click)

            self._speed_slider = self.server.gui.add_slider(
                "Speed",
                min=0.1,
                max=3.0,
                step=0.1,
                initial_value=1.0,
            )
            self._speed_slider.on_update(self._handle_speed_change)

            self._loop_checkbox = self.server.gui.add_checkbox(
                "Loop",
                initial_value=True,
            )
            self._loop_checkbox.on_update(self._handle_loop_change)

        # === Comparison ===
        with self.server.gui.add_folder("Reference Overlay"):
            self._show_reference = self.server.gui.add_checkbox(
                "Show Reference",
                initial_value=True,
            )
            self._show_reference.on_update(self._handle_reference_toggle)

            self._reference_opacity = self.server.gui.add_slider(
                "Reference Opacity",
                min=0.1,
                max=1.0,
                step=0.05,
                initial_value=0.3,
            )
            self._reference_opacity.on_update(self._handle_reference_opacity)

        # === Visibility ===
        with self.server.gui.add_folder("Visibility"):
            self._show_object = self.server.gui.add_checkbox(
                "Object",
                initial_value=True,
            )
            self._show_object.on_update(lambda _: self._handle_visibility("object", self._show_object.value))

            self._show_labels = self.server.gui.add_checkbox(
                "Frame Labels (Ghost)",
                initial_value=True,
            )
            self._show_labels.on_update(lambda _: self._handle_visibility("labels", self._show_labels.value))

            self._show_contacts = self.server.gui.add_checkbox(
                "Contact Points",
                initial_value=False,
            )
            self._show_contacts.on_update(lambda _: self._handle_visibility("contacts", self._show_contacts.value))

        # Setup playback frame change callback
        self.playback.set_on_frame_change(self._on_playback_frame_change)

    def set_trajectory_options(self, options: List[str], energies: Optional[List[float]] = None) -> None:
        """Update trajectory dropdown options.

        Args:
            options: List of trajectory names
            energies: Optional list of energy values
        """
        if not options:
            options = ["No trajectories loaded"]

        self._trajectory_dropdown.options = options
        self._trajectory_dropdown.value = options[0]

        # Update energy display
        if energies and len(energies) > 0:
            self._energy_text.value = f"{energies[0]:.4f}"
        else:
            self._energy_text.value = "N/A"

    def set_playback_info(self, n_frames: int, dt: float) -> None:
        """Update playback info.

        Args:
            n_frames: Number of frames
            dt: Time step
        """
        self.playback.set_trajectory_info(n_frames, dt)
        self._frame_slider.max = max(0, n_frames - 1)
        self._update_time_display()

    def _update_time_display(self) -> None:
        """Update the time display text."""
        current = self.playback.current_time
        total = self.playback.total_time
        self._frame_text.value = f"{current:.2f}s / {total:.2f}s"

    # === Callback Setters ===

    def set_on_trajectory_change(self, callback: Callable[[int], None]) -> None:
        """Set callback for trajectory selection changes."""
        self._on_trajectory_change = callback

    def set_on_view_mode_change(self, callback: Callable[[ViewMode], None]) -> None:
        """Set callback for view mode changes."""
        self._on_view_mode_change = callback

    def set_on_reference_toggle(self, callback: Callable[[bool], None]) -> None:
        """Set callback for reference visibility toggle."""
        self._on_reference_toggle = callback

    def set_on_reference_opacity(self, callback: Callable[[float], None]) -> None:
        """Set callback for reference opacity changes."""
        self._on_reference_opacity = callback

    def set_on_visibility_change(self, callback: Callable[[str, bool], None]) -> None:
        """Set callback for visibility toggles."""
        self._on_visibility_change = callback

    # === Event Handlers ===

    def _handle_trajectory_change(self, event) -> None:
        """Handle trajectory dropdown change."""
        if self._on_trajectory_change:
            # Parse index from dropdown value
            try:
                idx = int(event.target.value.split(":")[0])
                self._on_trajectory_change(idx)
            except (ValueError, IndexError):
                pass

    def _handle_view_mode_change(self, event) -> None:
        """Handle view mode button click."""
        mode = ViewMode.GHOST if event.target.value == "Ghost" else ViewMode.PLAYBACK
        if self._on_view_mode_change:
            self._on_view_mode_change(mode)

    def _handle_frame_slider(self, event) -> None:
        """Handle frame slider change."""
        self.playback.set_frame(int(event.target.value))
        self._update_time_display()

    def _handle_play_click(self, event) -> None:
        """Handle play button click."""
        if self.playback.is_playing:
            self.playback.pause()
            self._play_button.name = "▶ Play"
        else:
            self.playback.play()
            self._play_button.name = "⏸ Pause"

    def _handle_stop_click(self, event) -> None:
        """Handle stop button click."""
        self.playback.stop()
        self._play_button.name = "▶ Play"
        self._frame_slider.value = 0
        self._update_time_display()

    def _handle_speed_change(self, event) -> None:
        """Handle speed slider change."""
        self.playback.set_speed(event.target.value)

    def _handle_loop_change(self, event) -> None:
        """Handle loop checkbox change."""
        self.playback.loop = event.target.value

    def _handle_reference_toggle(self, event) -> None:
        """Handle reference visibility toggle."""
        if self._on_reference_toggle:
            self._on_reference_toggle(event.target.value)

    def _handle_reference_opacity(self, event) -> None:
        """Handle reference opacity change."""
        if self._on_reference_opacity:
            self._on_reference_opacity(event.target.value)

    def _handle_visibility(self, item: str, visible: bool) -> None:
        """Handle visibility toggle."""
        if self._on_visibility_change:
            self._on_visibility_change(item, visible)

    def _on_playback_frame_change(self, frame: int) -> None:
        """Handle frame changes from playback controller."""
        self._frame_slider.value = frame
        self._update_time_display()

    def update(self) -> None:
        """Update playback. Call this in the main loop."""
        self.playback.update()

        # Update play button state
        if self.playback.is_playing:
            self._play_button.name = "⏸ Pause"
        else:
            self._play_button.name = "▶ Play"
