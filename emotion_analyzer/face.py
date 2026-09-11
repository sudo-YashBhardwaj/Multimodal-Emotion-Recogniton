"""Face channel: MTCNN face detection + DeepFace expression classification."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN


@dataclass(frozen=True)
class Face:
    box: tuple[int, int, int, int]  # x, y, width, height
    emotion: str
    score: float  # classifier probability of `emotion`, 0-1
    confidence: float  # detector confidence, 0-1

    @property
    def area(self) -> int:
        return self.box[2] * self.box[3]


class FaceEmotionAnalyzer:
    def __init__(self, detection_threshold: float = 0.9) -> None:
        self._detector = MTCNN()
        self._threshold = detection_threshold

    def analyze(self, frame: np.ndarray) -> list[Face]:
        """Detect faces in a BGR frame and classify each one's expression."""
        height, width = frame.shape[:2]
        faces = []
        for detection in self._detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)):
            if detection["confidence"] < self._threshold:
                continue
            x, y, w, h = detection["box"]
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(width, int(x + w)), min(height, int(y + h))
            if x1 <= x0 or y1 <= y0:
                continue
            emotion, score = self._classify(frame[y0:y1, x0:x1])
            faces.append(Face((x0, y0, x1 - x0, y1 - y0), emotion, score, float(detection["confidence"])))
        return faces

    @staticmethod
    def _classify(crop: np.ndarray) -> tuple[str, float]:
        # The crop is already a face, so skip DeepFace's own detector.
        result = DeepFace.analyze(
            crop,
            actions=["emotion"],
            detector_backend="skip",
            enforce_detection=False,
            silent=True,
        )[0]
        emotion = result["dominant_emotion"]
        return emotion, float(result["emotion"][emotion]) / 100
