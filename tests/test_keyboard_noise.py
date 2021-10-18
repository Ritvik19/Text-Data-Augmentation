import json
import os

from text_data_augmentation import KeyBoardNoise


def test_keyboard_noise():
    data = json.load(open(os.path.join("tests", "test_data.json")))
    aug = KeyBoardNoise()
    aug(data)
