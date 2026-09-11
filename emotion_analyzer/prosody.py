"""Tone of voice from simple acoustic features.

A lightweight, training-free baseline: it measures loudness (RMS), zero-crossing
rate and spectral centroid of a speech window and maps them to an emotion with
hand-tuned thresholds. Cheap and dependency-free, but far less accurate than a
learned speech-emotion model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SILENCE_RMS = 0.01


@dataclass(frozen=True)
class ProsodyFeatures:
    rms: float  # loudness, for audio in [-1, 1]
    zcr: float  # zero crossings per sample
    centroid: float  # spectral centroid in Hz ("brightness")


def extract_features(audio: np.ndarray, sample_rate: int) -> ProsodyFeatures:
    audio = np.asarray(audio, dtype=np.float32).ravel()
    rms = float(np.sqrt(np.mean(audio**2)))
    zcr = float(np.mean(np.signbit(audio[1:]) != np.signbit(audio[:-1])))

    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / sample_rate)
    total = spectrum.sum()
    centroid = float((freqs * spectrum).sum() / total) if total > 0 else 0.0

    return ProsodyFeatures(rms, zcr, centroid)


def classify_features(f: ProsodyFeatures) -> str | None:
    """Map features to an emotion, or None when the window is silent."""
    if f.rms < SILENCE_RMS:
        return None
    if f.rms > 0.15:  # very animated
        if f.zcr > 0.15:
            return "happy"
        return "angry" if f.centroid > 2000 else "surprise"
    if f.rms > 0.08:  # lively
        return "happy" if f.zcr > 0.10 else "neutral"
    if f.rms < 0.02:  # quiet
        return "sad" if f.centroid < 1000 else "neutral"
    return "neutral"


def classify_tone(audio: np.ndarray, sample_rate: int) -> str | None:
    """Emotion conveyed by the tone of `audio`, or None if there is no speech."""
    if audio.size < sample_rate // 10:  # under 100 ms is too short to judge
        return None
    return classify_features(extract_features(audio, sample_rate))
