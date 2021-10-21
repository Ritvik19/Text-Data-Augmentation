import random

from tqdm.auto import tqdm
from transformers import pipeline


class AbstractiveSummarization:
    """AbstractiveSummarization Augmentation summarizes the model using transformer models.

    Args:
        model (string, optional): Transformer model. Defaults to sshleifer/distilbart-cnn-12-6
        show_progress (bool, optional): Set True to display progress bar.
            Defaults to True.
    """

    def __init__(self, model=None, show_progress=True):
        self.model = pipeline("summarization", model=model)
        self.disable_progress = not show_progress

    def __call__(self, x):
        augmented = []
        for sentence in tqdm(x, disable=self.disable_progress):
            augmented.append(self.model(sentence)[0]["summary_text"])
        return list(x) + augmented
