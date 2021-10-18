import json
import os

from text_data_augmentation import OCRNoise


def test_ocr_noise():
    data = json.load(open(os.path.join("tests", "test_data.json")))
    aug = OCRNoise()
    aug(data)
