from emotion_analyzer.emotions import EMOTIONS, GO_EMOTIONS_TO_EKMAN

GO_EMOTIONS = """
    admiration amusement anger annoyance approval caring confusion curiosity desire
    disappointment disapproval disgust embarrassment excitement fear gratitude grief
    joy love nervousness optimism pride realization relief remorse sadness surprise
    neutral
""".split()


def test_every_go_emotions_label_is_mapped():
    assert sorted(GO_EMOTIONS_TO_EKMAN) == sorted(GO_EMOTIONS)


def test_mapping_covers_the_shared_vocabulary():
    assert set(GO_EMOTIONS_TO_EKMAN.values()) == set(EMOTIONS)
