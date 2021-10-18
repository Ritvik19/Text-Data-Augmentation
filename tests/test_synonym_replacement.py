import json
import os

from text_data_augmentation import SynonymReplacement


def test_synonym_replacement():
    data = json.load(open(os.path.join("tests", "test_data.json")))
    aug = SynonymReplacement()
    aug(data)
