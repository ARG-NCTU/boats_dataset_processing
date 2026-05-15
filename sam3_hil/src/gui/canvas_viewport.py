"""Viewport geometry helpers for the annotation canvas."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CanvasViewportTransform:
    """Map between widget coordinates and original image coordinates."""

    widget_width: int
    widget_height: int
    image_width: int
    image_height: int
    zoom_factor: float = 1.0
    pan_x: int = 0
    pan_y: int = 0

    @property
    def fit_scale(self) -> float:
        if self.image_width <= 0 or self.image_height <= 0:
            return 1.0
        return min(
            self.widget_width / self.image_width,
            self.widget_height / self.image_height,
        )

    @property
    def display_scale(self) -> float:
        return max(0.01, self.fit_scale * self.zoom_factor)

    @property
    def scaled_size(self) -> Tuple[int, int]:
        return (
            max(1, int(self.image_width * self.display_scale)),
            max(1, int(self.image_height * self.display_scale)),
        )

    @property
    def image_offset(self) -> Tuple[int, int]:
        scaled_width, scaled_height = self.scaled_size
        return (
            (self.widget_width - scaled_width) // 2 + self.pan_x,
            (self.widget_height - scaled_height) // 2 + self.pan_y,
        )

    def screen_to_image(self, screen_x: int, screen_y: int) -> Tuple[int, int]:
        offset_x, offset_y = self.image_offset
        image_x = int((screen_x - offset_x) / self.display_scale)
        image_y = int((screen_y - offset_y) / self.display_scale)

        return (
            max(0, min(image_x, self.image_width - 1)),
            max(0, min(image_y, self.image_height - 1)),
        )

    def image_to_screen(self, image_x: int, image_y: int) -> Tuple[int, int]:
        offset_x, offset_y = self.image_offset
        return (
            int(image_x * self.display_scale) + offset_x,
            int(image_y * self.display_scale) + offset_y,
        )
