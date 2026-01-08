"""Playback controller for trajectory animation."""

import time
from enum import Enum, auto
from typing import Callable, Optional


class PlaybackState(Enum):
    """Playback state."""

    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()


class PlaybackController:
    """Controls trajectory playback/animation."""

    def __init__(self, n_frames: int = 1, dt: float = 0.01):
        """Initialize playback controller.

        Args:
            n_frames: Number of frames
            dt: Time step in seconds
        """
        self.n_frames = n_frames
        self.dt = dt
        self.current_frame = 0
        self.state = PlaybackState.STOPPED
        self.speed = 1.0
        self.loop = True

        # Timing
        self._last_update_time = 0.0
        self._accumulated_time = 0.0

        # Callback for frame changes
        self._on_frame_change: Optional[Callable[[int], None]] = None

    def set_trajectory_info(self, n_frames: int, dt: float) -> None:
        """Update trajectory information.

        Args:
            n_frames: Number of frames
            dt: Time step
        """
        self.n_frames = max(1, n_frames)
        self.dt = dt
        self.current_frame = min(self.current_frame, self.n_frames - 1)

    def set_on_frame_change(self, callback: Callable[[int], None]) -> None:
        """Set callback for frame changes.

        Args:
            callback: Function called with new frame index
        """
        self._on_frame_change = callback

    def play(self) -> None:
        """Start playback."""
        if self.n_frames <= 1:
            return
        self.state = PlaybackState.PLAYING
        self._last_update_time = time.time()
        self._accumulated_time = 0.0

    def pause(self) -> None:
        """Pause playback."""
        self.state = PlaybackState.PAUSED

    def stop(self) -> None:
        """Stop and reset to frame 0."""
        self.state = PlaybackState.STOPPED
        self.set_frame(0)

    def toggle_play(self) -> None:
        """Toggle between play and pause."""
        if self.state == PlaybackState.PLAYING:
            self.pause()
        else:
            self.play()

    def set_frame(self, frame: int) -> None:
        """Set current frame.

        Args:
            frame: Frame index
        """
        frame = max(0, min(frame, self.n_frames - 1))
        if frame != self.current_frame:
            self.current_frame = frame
            if self._on_frame_change:
                self._on_frame_change(frame)

    def step_forward(self) -> None:
        """Step one frame forward."""
        next_frame = self.current_frame + 1
        if next_frame >= self.n_frames:
            if self.loop:
                next_frame = 0
            else:
                next_frame = self.n_frames - 1
        self.set_frame(next_frame)

    def step_backward(self) -> None:
        """Step one frame backward."""
        prev_frame = self.current_frame - 1
        if prev_frame < 0:
            if self.loop:
                prev_frame = self.n_frames - 1
            else:
                prev_frame = 0
        self.set_frame(prev_frame)

    def go_to_start(self) -> None:
        """Go to first frame."""
        self.set_frame(0)

    def go_to_end(self) -> None:
        """Go to last frame."""
        self.set_frame(self.n_frames - 1)

    def set_speed(self, speed: float) -> None:
        """Set playback speed multiplier.

        Args:
            speed: Speed multiplier (1.0 = normal)
        """
        self.speed = max(0.1, min(5.0, speed))

    def update(self) -> None:
        """Update playback state. Call this in the main loop."""
        if self.state != PlaybackState.PLAYING or self.n_frames <= 1:
            return

        current_time = time.time()
        elapsed = current_time - self._last_update_time
        self._last_update_time = current_time

        # Accumulate time
        self._accumulated_time += elapsed * self.speed

        # Check if we should advance frames
        while self._accumulated_time >= self.dt:
            self._accumulated_time -= self.dt

            next_frame = self.current_frame + 1
            if next_frame >= self.n_frames:
                if self.loop:
                    next_frame = 0
                else:
                    next_frame = self.n_frames - 1
                    self.state = PlaybackState.STOPPED
                    break

            self.current_frame = next_frame
            if self._on_frame_change:
                self._on_frame_change(next_frame)

    @property
    def is_playing(self) -> bool:
        """Whether currently playing."""
        return self.state == PlaybackState.PLAYING

    @property
    def progress(self) -> float:
        """Current progress (0.0 - 1.0)."""
        if self.n_frames <= 1:
            return 1.0
        return self.current_frame / (self.n_frames - 1)

    @property
    def current_time(self) -> float:
        """Current time in seconds."""
        return self.current_frame * self.dt

    @property
    def total_time(self) -> float:
        """Total trajectory time in seconds."""
        return (self.n_frames - 1) * self.dt
