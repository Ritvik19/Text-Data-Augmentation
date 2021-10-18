import json
import os

from text_data_augmentation import CharacterNoise


def test_character_noise():
    data = json.load(open(os.path.join("tests", "test_data.json")))
    aug = CharacterNoise()
    aug(data)
