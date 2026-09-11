"""Session history and the end-of-session emotion report."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import groupby

from .fusion import FusedEmotion

NEGATIVE = ("angry", "disgust", "fear", "sad")
POSITIVE = ("happy", "surprise")


@dataclass(frozen=True)
class Period:
    emotion: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class EmotionTracker:
    """Records fused emotions over a session and summarises them."""

    def __init__(self) -> None:
        self.events: list[FusedEmotion] = []

    def add(self, event: FusedEmotion) -> None:
        self.events.append(event)

    def distribution(self) -> Counter:
        return Counter(event.emotion for event in self.events)

    def periods(self, emotions=NEGATIVE, min_duration: float = 3.0) -> list[Period]:
        """Uninterrupted runs of any of `emotions` lasting at least `min_duration` seconds."""
        periods = []
        for emotion, run in groupby(self.events, key=lambda event: event.emotion):
            run = list(run)
            period = Period(emotion, run[0].timestamp, run[-1].timestamp)
            if emotion in emotions and period.duration >= min_duration:
                periods.append(period)
        return periods

    def sentiment(self) -> str:
        if not self.events:
            return "unknown"
        counts, total = self.distribution(), len(self.events)
        positive = sum(counts[e] for e in POSITIVE) / total
        negative = sum(counts[e] for e in NEGATIVE) / total
        label = "positive" if positive > 0.4 else "negative" if negative > 0.4 else "neutral"
        return f"{label} ({positive:.0%} positive, {negative:.0%} negative)"

    def summary(self) -> str:
        if not self.events:
            return "No emotion data was collected."

        total = len(self.events)
        duration = self.events[-1].timestamp - self.events[0].timestamp
        lines = [
            "Emotion summary",
            "===============",
            f"Duration: {duration:.1f}s ({total} observations)",
            "",
            "Distribution",
        ]
        for emotion, count in self.distribution().most_common():
            share = count / total
            lines.append(f"  {emotion:<9}{share:>5.0%}  {'█' * round(share * 30)}")

        periods = [
            f"  {p.emotion:<9}{p.start:6.1f}s -> {p.end:.1f}s ({p.duration:.1f}s)"
            for p in self.periods()
        ]
        lines += ["", "Sustained negative emotion", *(periods or ["  none"])]
        lines += ["", f"Overall sentiment: {self.sentiment()}"]
        return "\n".join(lines)
