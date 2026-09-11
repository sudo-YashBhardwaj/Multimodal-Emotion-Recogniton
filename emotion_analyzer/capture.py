"""Frame and audio capture from a webcam + microphone or a media file."""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from collections import deque

import cv2
import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000  # Whisper's native rate; all audio is captured/decoded at it.
_MIC_BLOCK = 1024
_MIC_BUFFER_SECONDS = 30  # Whisper's maximum context


class MediaSource:
    """Streams video frames on a background thread and serves audio windows on demand.

    Live sources (a webcam index) record the default microphone into a rolling
    buffer; file sources decode the whole audio track up front with ffmpeg.
    Timestamps are seconds since the start of the stream.
    """

    def __init__(self, source: int | str, *, audio: bool = True) -> None:
        self.source = source
        self.is_live = isinstance(source, int)
        self._audio = audio
        self._frames: queue.Queue = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._done = threading.Event()
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

        self._track: np.ndarray | None = None  # file audio
        self._mic: deque = deque(maxlen=_MIC_BUFFER_SECONDS * SAMPLE_RATE // _MIC_BLOCK)
        self._mic_lock = threading.Lock()
        self._stream = None

    def __enter__(self) -> MediaSource:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        self._capture = cv2.VideoCapture(self.source)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source!r}")

        if self._audio:
            if self.is_live:
                self._start_microphone()
            else:
                self._track = load_audio(self.source)

        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._read_frames, name="capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._stream:
            self._stream.stop()
            self._stream.close()
        if self._capture:
            self._capture.release()

    @property
    def exhausted(self) -> bool:
        """True once the source has ended and every frame has been consumed."""
        return self._done.is_set() and self._frames.empty()

    def read(self, timeout: float = 0.1) -> tuple[np.ndarray, float] | None:
        """Next `(frame, timestamp)`, or None if none arrived within `timeout`."""
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def audio_window(self, end: float, duration: float) -> np.ndarray | None:
        """Up to `duration` seconds of mono 16 kHz audio ending at stream time `end`.

        Live sources always return the most recent audio.
        """
        n = int(duration * SAMPLE_RATE)
        if self.is_live:
            with self._mic_lock:
                chunks = list(self._mic)
            return np.concatenate(chunks)[-n:] if chunks else None

        if self._track is None:
            return None
        stop = min(int(end * SAMPLE_RATE), self._track.size)
        start = max(0, stop - n)
        return self._track[start:stop] if stop > start else None

    def _read_frames(self) -> None:
        fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        index = 0
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if not ok:
                break
            if self.is_live:
                # Never stall the camera: drop the frame if analysis is behind.
                try:
                    self._frames.put_nowait((frame, time.monotonic() - self._t0))
                except queue.Full:
                    pass
            else:
                # Files: keep every frame and wait for the consumer instead.
                item = (frame, index / fps)
                index += 1
                while not self._stop.is_set():
                    try:
                        self._frames.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        self._done.set()

    def _start_microphone(self) -> None:
        import sounddevice as sd  # needs PortAudio, so only imported for live capture

        def on_audio(indata, frames, time_info, status):
            if status:
                log.warning("Microphone: %s", status)
            with self._mic_lock:
                self._mic.append(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=_MIC_BLOCK,
            callback=on_audio,
        )
        self._stream.start()


def load_audio(path: str) -> np.ndarray | None:
    """Decode a file's audio track to mono 16 kHz float32, or None if it has none."""
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", path,
        "-vn", "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
    ]  # fmt: skip
    try:
        pcm = subprocess.run(cmd, capture_output=True).stdout
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is required to read audio from video files") from None
    if not pcm:
        log.warning("No audio track in %s, continuing with video only", path)
        return None
    return np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
