"""Text channel: emotion of transcribed speech with a GoEmotions RoBERTa model."""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .emotions import GO_EMOTIONS_TO_EKMAN

DEFAULT_TEXT_MODEL = "SamLowe/roberta-base-go_emotions"


class TextEmotionClassifier:
    def __init__(self, model_name: str = DEFAULT_TEXT_MODEL, device: str = "cpu") -> None:
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()

    def classify(self, text: str) -> tuple[str, float] | None:
        """Dominant emotion of `text` and its score, or None for empty text."""
        if not text.strip():
            return None

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True).to(self._device)
        with torch.inference_mode():
            # The model is multi-label, so each label gets an independent sigmoid.
            probs = torch.sigmoid(self._model(**inputs).logits[0]).tolist()

        # Collapse the 28 GoEmotions labels onto the shared 7-class vocabulary.
        scores: dict[str, float] = {}
        for index, prob in enumerate(probs):
            emotion = GO_EMOTIONS_TO_EKMAN[self._model.config.id2label[index]]
            scores[emotion] = max(scores.get(emotion, 0.0), prob)

        emotion = max(scores, key=scores.get)
        return emotion, scores[emotion]
