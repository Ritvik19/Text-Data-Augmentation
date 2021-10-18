import json
import os

from text_data_augmentation import SimilarWordReplacement


def test_similar_word_replacement():
    data = json.load(open(os.path.join("tests", "test_data.json")))[:10]
    aug = SimilarWordReplacement("en_core_web_lg")
    aug(data)
