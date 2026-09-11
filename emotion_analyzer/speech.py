"""Speech channels: tone of voice + Whisper transcription, off the UI thread."""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass

import numpy as np
import whisper

from .capture import SAMPLE_RATE
from .prosody import classify_tone
from .text import DEFAULT_TEXT_MODEL, TextEmotionClassifier

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeechResult:
    timestamp: float  # stream time at the end of the analysed window
    transcript: str
    tone: str | None  # emotion from the voice (None = silence)
    text_emotion: str | None  # emotion from the words (None = nothing said)


class SpeechAnalyzer:
    def __init__(
        self,
        whisper_model: str = "base.en",
        text_model: str = DEFAULT_TEXT_MODEL,
        device: str = "cpu",
    ) -> None:
        self._whisper = whisper.load_model(whisper_model, device=device)
        self._fp16 = device == "cuda"
        self._text = TextEmotionClassifier(text_model, device)

    def analyze(self, audio: np.ndarray, timestamp: float) -> SpeechResult:
        """Analyse a window of mono 16 kHz float32 audio."""
        tone = classify_tone(audio, SAMPLE_RATE)
        if tone is None:  # silence: skip Whisper, which tends to hallucinate on it
            return SpeechResult(timestamp, "", None, None)

        transcript = self._whisper.transcribe(
            audio,
            language="en",
            fp16=self._fp16,
            condition_on_previous_text=False,
        )["text"].strip()
        classified = self._text.classify(transcript)
        return SpeechResult(timestamp, transcript, tone, classified[0] if classified else None)


class SpeechWorker:
    """Runs a SpeechAnalyzer on a background thread so video never waits on Whisper.

    Windows submitted while the worker is busy are dropped; `latest` always holds
    the most recent result.
    """

    def __init__(self, analyzer: SpeechAnalyzer) -> None:
        self._analyzer = analyzer
        self._inbox: queue.Queue = queue.Queue(maxsize=1)
        self.latest: SpeechResult | None = None
        threading.Thread(target=self._run, name="speech", daemon=True).start()

    def submit(self, audio: np.ndarray, timestamp: float) -> None:
        try:
            self._inbox.put_nowait((audio, timestamp))
        except queue.Full:
            pass

    def _run(self) -> None:
        while True:
            audio, timestamp = self._inbox.get()
            try:
                self.latest = self._analyzer.analyze(audio, timestamp)
            except Exception:
                log.exception("Speech analysis failed")
