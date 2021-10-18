import json
import os

from text_data_augmentation import BackTranslation


def test_back_translation():
    data = json.load(open(os.path.join("tests", "test_data.json")))
    aug = BackTranslation(base_language="en", interim_language="es")
    aug(data)
