import json
import os

from text_data_augmentation import EasyDataAugmentation


def test_easy_data_augmentation():
    data = json.load(open(os.path.join("tests", "test_data.json")))
    aug = EasyDataAugmentation()
    aug(data)
