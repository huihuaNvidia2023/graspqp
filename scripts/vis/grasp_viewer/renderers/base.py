"""Base renderer interface."""

from abc import ABC, abstractmethod
from typing import Tuple

import viser


class BaseRenderer(ABC):
    """Abstract base class for renderers."""

    def __init__(self, server: viser.ViserServer, name: str):
        """Initialize renderer.

        Args:
            server: Viser server instance
            name: Unique name for this renderer (used in scene graph)
        """
        self.server = server
        self.name = name
        self._visible = True
        self._opacity = 1.0
        self._color: Tuple[int, int, int] = (180, 180, 180)

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update the renderer with new data."""
        pass

    @abstractmethod
    def remove(self) -> None:
        """Remove all scene elements created by this renderer."""
        pass

    @property
    def visible(self) -> bool:
        """Whether the renderer is visible."""
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        """Set visibility."""
        self._visible = value
        self._update_visibility()

    @property
    def opacity(self) -> float:
        """Current opacity (0.0 - 1.0)."""
        return self._opacity

    @opacity.setter
    def opacity(self, value: float) -> None:
        """Set opacity."""
        self._opacity = max(0.0, min(1.0, value))
        self._update_appearance()

    @property
    def color(self) -> Tuple[int, int, int]:
        """Current RGB color (0-255)."""
        return self._color

    @color.setter
    def color(self, value: Tuple[int, int, int]) -> None:
        """Set RGB color."""
        self._color = value
        self._update_appearance()

    def _update_visibility(self) -> None:
        """Update visibility of all elements. Override in subclass."""
        pass

    def _update_appearance(self) -> None:
        """Update color/opacity of all elements. Override in subclass."""
        pass

    def get_rgba(self) -> Tuple[int, int, int, int]:
        """Get RGBA color with current opacity."""
        alpha = int(self._opacity * 255)
        return (*self._color, alpha)
