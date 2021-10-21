import json
import os

from text_data_augmentation import AbstractiveSummarization


def test_back_translation():
    data = json.load(open(os.path.join("tests", "test_data.json")))[:10]
    aug = AbstractiveSummarization()
    aug(data)
