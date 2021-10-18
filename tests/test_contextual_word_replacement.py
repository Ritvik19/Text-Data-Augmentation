import json
import os

from text_data_augmentation import ContextualWordReplacement


def test_contextual_word_replacement():
    data = json.load(open(os.path.join("tests", "test_data.json")))[:10]
    aug = ContextualWordReplacement()
    aug(data)
