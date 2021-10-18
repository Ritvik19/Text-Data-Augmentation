import json
import os

from text_data_augmentation import WordSplit


def test_word_split():
    data = json.load(open(os.path.join("tests", "test_data.json")))
    aug = WordSplit()
    aug(data)
