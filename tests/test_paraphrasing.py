import json
import os

from text_data_augmentation import Paraphrase


def test_paraphrasing():
    data = json.load(open(os.path.join("tests", "test_data.json")))[:10]
    aug = Paraphrase("hetpandya/t5-small-tapaco")
    aug(data)
    aug = Paraphrase("Vamsi/T5_Paraphrase_Paws")
    aug(data)
