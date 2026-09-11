"""Late fusion of per-modality emotions into a single estimate."""

from __future__ import annotations

from dataclasses import dataclass

# Most to least trusted: what is said, then the face, then the tone of voice.
PRIORITY = ("text", "face", "voice")


@dataclass(frozen=True)
class FusedEmotion:
    timestamp: float
    emotion: str
    confidence: float
    sources: dict[str, str | None]  # per-modality emotion, None = no signal


def fuse(
    timestamp: float,
    *,
    face: str | None = None,
    voice: str | None = None,
    text: str | None = None,
) -> FusedEmotion | None:
    """Combine the emotions reported by each modality (None = no signal).

    The highest-priority modality with a non-neutral emotion wins, so a clear
    signal on any channel beats "neutral" on the others. Confidence grows with
    the number of modalities that agree with the winner: 0.5 + 0.4 * agreeing / 3,
    i.e. ~0.63 for a single channel up to 0.9 when all three agree.

    Returns None when no modality has a signal.
    """
    sources = {"face": face, "voice": voice, "text": text}
    present = [sources[m] for m in PRIORITY if sources[m] is not None]
    if not present:
        return None

    emotion = next((e for e in present if e != "neutral"), "neutral")
    confidence = 0.5 + 0.4 * present.count(emotion) / len(PRIORITY)
    return FusedEmotion(timestamp, emotion, confidence, sources)
