import pytest

from emotion_analyzer.fusion import fuse


def test_no_signal_gives_no_estimate():
    assert fuse(0.0) is None


def test_non_neutral_beats_neutral():
    assert fuse(0.0, face="neutral", voice="angry", text="neutral").emotion == "angry"


def test_all_neutral_is_neutral():
    assert fuse(0.0, face="neutral", voice="neutral").emotion == "neutral"


@pytest.mark.parametrize(
    "signals, expected",
    [
        ({"face": "sad", "voice": "angry", "text": "happy"}, "happy"),
        ({"face": "sad", "voice": "angry"}, "sad"),
        ({"voice": "angry"}, "angry"),
    ],
)
def test_priority_is_text_then_face_then_voice(signals, expected):
    assert fuse(0.0, **signals).emotion == expected


def test_confidence_grows_with_agreement():
    one = fuse(0.0, face="happy", voice="neutral")
    two = fuse(0.0, face="happy", voice="neutral", text="happy")
    three = fuse(0.0, face="happy", voice="happy", text="happy")
    assert one.confidence < two.confidence < three.confidence
    assert three.confidence == pytest.approx(0.9)


def test_reports_sources_and_timestamp():
    result = fuse(12.5, face="fear")
    assert result.timestamp == 12.5
    assert result.sources == {"face": "fear", "voice": None, "text": None}
