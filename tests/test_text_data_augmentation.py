import json
import os

from text_data_augmentation import __version__


def test_version():
    assert __version__ == "0.12.0"


def test_data():
    json.load(open(os.path.join("tests", "test_data.json")))
