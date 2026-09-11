"""Live OpenCV overlay: face boxes, fused-emotion HUD, transcript and legend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from .emotions import EMOTIONS
from .fusion import FusedEmotion

if TYPE_CHECKING:
    from .face import Face

# BGR
COLORS = {
    "angry": (60, 60, 230),
    "disgust": (40, 140, 240),
    "fear": (40, 210, 240),
    "happy": (90, 200, 90),
    "sad": (210, 130, 50),
    "surprise": (200, 90, 200),
    "neutral": (170, 170, 170),
}
WHITE = (245, 245, 245)
MUTED = (150, 150, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Display:
    def __init__(self) -> None:
        self.show_legend = False

    def render(
        self,
        frame: np.ndarray,
        faces: list[Face],
        fused: FusedEmotion | None,
        transcript: str = "",
    ) -> np.ndarray:
        canvas = frame.copy()
        _draw_hud(canvas, fused)
        for face in faces:  # drawn after the HUD so boxes are never dimmed by it
            _draw_face(canvas, face)
        if transcript:
            _draw_transcript(canvas, transcript)
        if self.show_legend:
            _draw_legend(canvas)
        return canvas


def _draw_face(canvas: np.ndarray, face: Face) -> None:
    x, y, w, h = face.box
    color = COLORS.get(face.emotion, WHITE)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)

    label = f"{face.emotion} {face.score:.0%}"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
    top = y - th - 10 if y >= th + 10 else y + h  # keep the label on-screen
    cv2.rectangle(canvas, (x, top), (x + tw + 8, top + th + 10), color, cv2.FILLED)
    _text(canvas, label, (x + 4, top + th + 5), 0.55, _ink(color))


def _draw_hud(canvas: np.ndarray, fused: FusedEmotion | None) -> None:
    _panel(canvas, 10, 10, 250, 122)
    if fused is None:
        _text(canvas, "no signal", (22, 44), 0.8, MUTED, 2)
        return

    _text(canvas, fused.emotion.upper(), (22, 44), 0.9, COLORS.get(fused.emotion, WHITE), 2)
    _text(canvas, f"{fused.confidence:.0%}", (190, 44), 0.6, MUTED)
    for row, (modality, emotion) in enumerate(fused.sources.items()):
        y = 72 + row * 20
        _text(canvas, modality, (22, y), 0.5, MUTED)
        _text(canvas, emotion or "-", (90, y), 0.5, COLORS.get(emotion, MUTED))


def _draw_transcript(canvas: np.ndarray, transcript: str) -> None:
    height, width = canvas.shape[:2]
    _panel(canvas, 10, height - 42, width - 10, height - 10)
    _text(canvas, _fit(f'"{transcript}"', width - 40, 0.55), (20, height - 20), 0.55)


def _draw_legend(canvas: np.ndarray) -> None:
    height, width = canvas.shape[:2]
    x0, y1 = width - 140, height - 52
    y0 = y1 - 20 * len(EMOTIONS) - 16
    _panel(canvas, x0, y0, width - 10, y1)
    for row, emotion in enumerate(EMOTIONS):
        y = y0 + 22 + 20 * row
        cv2.rectangle(canvas, (x0 + 10, y - 10), (x0 + 22, y + 2), COLORS[emotion], cv2.FILLED)
        _text(canvas, emotion, (x0 + 30, y), 0.45)


def _panel(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int, alpha: float = 0.6) -> None:
    """Darken a rectangle so text drawn on it stays readable."""
    height, width = canvas.shape[:2]
    region = canvas[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)]
    region[:] = (region * (1 - alpha)).astype(np.uint8)


def _text(canvas, text, origin, scale=0.5, color=WHITE, thickness=1) -> None:
    cv2.putText(canvas, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def _fit(text: str, max_width: int, scale: float) -> str:
    """Trim `text` from the left, keeping the most recent words, until it fits."""
    def width(s: str) -> int:
        return cv2.getTextSize(s, FONT, scale, 1)[0][0]

    if width(text) <= max_width:
        return text
    while text and width("..." + text) > max_width:
        text = text[1:]
    return "..." + text


def _ink(background) -> tuple[int, int, int]:
    """Dark or light text, whichever reads better on a BGR `background`."""
    b, g, r = background
    return (20, 20, 20) if 0.114 * b + 0.587 * g + 0.299 * r > 140 else WHITE
