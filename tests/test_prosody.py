import numpy as np
import pytest

from emotion_analyzer.prosody import ProsodyFeatures, classify_features, classify_tone, extract_features

SR = 16_000


def sine(freq: float, amplitude: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_features_of_a_pure_tone():
    features = extract_features(sine(440, 0.5), SR)
    assert features.rms == pytest.approx(0.5 / np.sqrt(2), rel=1e-3)
    assert features.zcr == pytest.approx(2 * 440 / SR, rel=0.02)
    assert features.centroid == pytest.approx(440, rel=0.05)


def test_silence_and_tiny_windows_have_no_tone():
    assert classify_tone(np.zeros(SR, dtype=np.float32), SR) is None
    assert classify_tone(sine(440, 0.5, seconds=0.05), SR) is None


@pytest.mark.parametrize(
    "rms, zcr, centroid, expected",
    [
        (0.005, 0.10, 1500, None),
        (0.20, 0.20, 1500, "happy"),
        (0.20, 0.05, 3000, "angry"),
        (0.20, 0.05, 1000, "surprise"),
        (0.10, 0.12, 1500, "happy"),
        (0.10, 0.05, 1500, "neutral"),
        (0.015, 0.05, 500, "sad"),
        (0.015, 0.05, 1500, "neutral"),
        (0.05, 0.05, 1500, "neutral"),
    ],
)
def test_rules(rms, zcr, centroid, expected):
    assert classify_features(ProsodyFeatures(rms, zcr, centroid)) == expected
