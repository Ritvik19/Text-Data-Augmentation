import random

from nltk import sent_tokenize
from tqdm.auto import tqdm
from transformers import pipeline


class ContextualWordReplacement:
    """Contextual Word Replacement augmentation creates Augmented Samples by
    randomly replacing some words with a mask and then using a Masked Language
    Model to fill it.

    Args:
        model (string): Transformer Model to generate vectors. Defaults to distil-roberta.
        n_aug (int, optional): Number of augmentations to be created for one sentence.
            Defaults to 10.
        seed (int, optional): Random State for reproducibility. Defaults to None.
        show_progress (bool, optional): Set True to display progress bar.
            Defaults to True.
    """

    def __init__(self, model=None, n_aug=10, seed=None, show_progress=True):
        self.model = pipeline("fill-mask", model=model)
        self.n_aug = n_aug
        self.seed = seed
        self.disable_progress = not show_progress

    def __contextual_word_replacement(self, sentence):
        words = sentence.split()
        r_idx = random.randint(0, len(words) - 1)
        words[r_idx] = "<mask>"
        masked_sentence = " ".join(words)
        try:
            augmentations = [s["sequence"] for s in self.model(masked_sentence)]
            return random.choice(augmentations)
        except RuntimeError:
            return sentence

    def __call__(self, x):
        random.seed(self.seed)
        augmented = []
        for sentence in tqdm(x, disable=self.disable_progress):
            for _ in range(self.n_aug):
                augmented.append(
                    " ".join(
                        [
                            self.__contextual_word_replacement(sent)
                            for sent in sent_tokenize(sentence)
                        ]
                    )
                )
        return list(x) + augmented
