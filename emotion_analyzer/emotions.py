"""The emotion vocabulary shared by every modality.

All channels report in the same seven classes (Ekman's six basic emotions plus
neutral), which is also what DeepFace's facial-expression model emits.
"""

EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

# GoEmotions' 27 emotions + neutral, grouped by the Ekman mapping published
# with the dataset (Demszky et al., 2020).
GO_EMOTIONS_TO_EKMAN = {
    "anger": "angry",
    "annoyance": "angry",
    "disapproval": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "nervousness": "fear",
    "admiration": "happy",
    "amusement": "happy",
    "approval": "happy",
    "caring": "happy",
    "desire": "happy",
    "excitement": "happy",
    "gratitude": "happy",
    "joy": "happy",
    "love": "happy",
    "optimism": "happy",
    "pride": "happy",
    "relief": "happy",
    "disappointment": "sad",
    "embarrassment": "sad",
    "grief": "sad",
    "remorse": "sad",
    "sadness": "sad",
    "confusion": "surprise",
    "curiosity": "surprise",
    "realization": "surprise",
    "surprise": "surprise",
    "neutral": "neutral",
}
