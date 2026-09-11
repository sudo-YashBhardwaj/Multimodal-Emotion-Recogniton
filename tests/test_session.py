from emotion_analyzer.fusion import FusedEmotion
from emotion_analyzer.session import EmotionTracker, Period


def tracker_with(*emotions: str, step: float = 1.0) -> EmotionTracker:
    tracker = EmotionTracker()
    for i, emotion in enumerate(emotions):
        tracker.add(FusedEmotion(i * step, emotion, 0.7, {}))
    return tracker


def test_distribution_counts_each_emotion():
    tracker = tracker_with("happy", "happy", "sad")
    assert tracker.distribution() == {"happy": 2, "sad": 1}


def test_periods_find_sustained_negative_runs():
    tracker = tracker_with("angry", "angry", "angry", "angry", "happy", "sad", "sad")
    assert tracker.periods(min_duration=3.0) == [Period("angry", 0.0, 3.0)]


def test_periods_ignore_short_and_positive_runs():
    tracker = tracker_with("sad", "happy", "happy", "happy", "happy", "happy")
    assert tracker.periods(min_duration=3.0) == []


def test_sentiment():
    assert tracker_with("happy", "happy", "neutral").sentiment().startswith("positive")
    assert tracker_with("sad", "angry", "neutral").sentiment().startswith("negative")
    assert tracker_with("neutral", "neutral", "happy").sentiment().startswith("neutral")


def test_summary_mentions_duration_and_emotions():
    summary = tracker_with("happy", "happy", "sad").summary()
    assert "Duration: 2.0s (3 observations)" in summary
    assert "happy" in summary and "sad" in summary


def test_empty_session():
    tracker = EmotionTracker()
    assert tracker.summary() == "No emotion data was collected."
    assert tracker.sentiment() == "unknown"
