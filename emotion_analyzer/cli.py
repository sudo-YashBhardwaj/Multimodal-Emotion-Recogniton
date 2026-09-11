"""Command-line entry point: capture -> per-modality analysis -> fusion -> display."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import cv2

from .capture import MediaSource
from .fusion import fuse
from .session import EmotionTracker

log = logging.getLogger("emotion_analyzer")

WINDOW_TITLE = "Multimodal Emotion Recognition"


def parse_source(value: str) -> int | str:
    if value.isdigit():
        return int(value)
    if not os.path.isfile(value):
        raise argparse.ArgumentTypeError(f"video file not found: {value}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="emotion-recognition",
        description="Real-time emotion recognition from face, voice and speech.",
    )
    parser.add_argument(
        "--source", type=parse_source, default=0,
        help="webcam index or path to a video file (default: 0)",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu",
        help="device for the Whisper and text models (default: cpu)",
    )
    parser.add_argument(
        "--whisper-model", default="base.en",
        help="Whisper checkpoint, e.g. tiny.en, base.en, small.en (default: base.en)",
    )
    parser.add_argument(
        "--detection-threshold", type=float, default=0.9,
        help="minimum face-detector confidence, 0-1 (default: 0.9)",
    )
    parser.add_argument(
        "--audio-window", type=float, default=4.0,
        help="seconds of audio per speech analysis (default: 4)",
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="analyse facial expressions only",
    )  # fmt: skip
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    log.setLevel(logging.INFO)  # our messages only; keep third-party libraries at WARNING
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieten TensorFlow (DeepFace)

    # Deferred so that --help and argument errors don't wait on TensorFlow/PyTorch.
    from .face import FaceEmotionAnalyzer
    from .speech import SpeechAnalyzer, SpeechWorker

    log.info("Loading models (downloaded on first run)...")
    faces = FaceEmotionAnalyzer(args.detection_threshold)
    speech = None
    if not args.no_audio:
        speech = SpeechWorker(SpeechAnalyzer(args.whisper_model, device=args.device))

    tracker = EmotionTracker()
    try:
        with MediaSource(args.source, audio=speech is not None) as media:
            run(media, faces, speech, tracker, args.audio_window)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        log.error("%s", e)
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()
        print(f"\n{tracker.summary()}\n")


def run(media, faces, speech, tracker: EmotionTracker, window: float) -> None:
    from .display import Display

    display = Display()
    next_window = window
    log.info("Running. Keys: q quit | s summary | l legend")

    while not media.exhausted:
        item = media.read()
        if item is None:
            continue
        frame, t = item

        found = faces.analyze(frame)
        face = max(found, key=lambda f: f.area).emotion if found else None  # closest person

        heard = None
        if speech:
            if t >= next_window:
                audio = media.audio_window(end=t, duration=window)
                if audio is not None:
                    speech.submit(audio, t)
                next_window = t + window
            heard = speech.latest
            if heard and t - heard.timestamp > 2 * window:  # stale
                heard = None

        fused = fuse(
            t,
            face=face,
            voice=heard.tone if heard else None,
            text=heard.text_emotion if heard else None,
        )
        if fused:
            tracker.add(fused)

        cv2.imshow(WINDOW_TITLE, display.render(frame, found, fused, heard.transcript if heard else ""))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("s"):
            print(f"\n{tracker.summary()}\n")
        elif key == ord("l"):
            display.show_legend = not display.show_legend
