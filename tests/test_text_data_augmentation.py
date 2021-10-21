import json
import os

from text_data_augmentation import __version__


def test_version():
    assert __version__ == "1.0.1"


def test_data():
    json.load(open(os.path.join("tests", "test_data.json")))
