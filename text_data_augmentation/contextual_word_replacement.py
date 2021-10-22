import random
import re

import numpy as np
from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from transformers import pipeline
from transformers.pipelines import PipelineException


class ContextualWordReplacement:
    """Contextual Word Replacement augmentation creates Augmented Samples by
    randomly replacing some words with a mask and then using a Masked Language
    Model to fill it.

    Args:
        model (string): Transformer Model to generate vectors. Defaults to distil-roberta.
        n_aug (int, optional): Number of augmentations to be created for one sentence.
            Defaults to 10.
        use_tfidf (bool, optional): Whether to use TFIDF weights to sample a word in the
            sentence. Defaults to True.
        seed (int, optional): Random State for reproducibility. Defaults to None.
        show_progress (bool, optional): Set True to display progress bar.
            Defaults to True.
    """

    def __init__(
        self, model=None, n_aug=10, use_tfidf=True, seed=None, show_progress=True
    ):
        self.model = pipeline("fill-mask", model=model)
        self.n_aug = n_aug
        self.use_tfidf = use_tfidf
        self.seed = seed
        self.disable_progress = not show_progress

    def __contextual_word_replacement(self, sentence):
        if self.use_tfidf:
            v = self.tvec.transform([sentence])
            v = np.ravel(v.todense())
            c = np.max(v)
            z = np.sum(c - v) / np.mean(v)
            weights = np.where(
                v != 0, np.minimum((0.7 * (c - v) / z), np.ones_like(v)), 0
            )
            indices = np.arange(len(v))
            word_2_replace = self.tvec.get_feature_names()[
                random.choices(indices, weights=weights)[0]
            ]
            masked_sentence = re.sub(
                word_2_replace, "<mask>", sentence, 1, re.IGNORECASE
            )
        else:
            words = sentence.split()
            r_idx = random.randint(0, len(words) - 1)
            words[r_idx] = "<mask>"
            masked_sentence = " ".join(words)
        try:
            augmentations = [s["sequence"] for s in self.model(masked_sentence)]
            return random.choice(augmentations)
        except (RuntimeError, PipelineException):
            return sentence

    def __call__(self, x):
        random.seed(self.seed)
        if self.use_tfidf:
            self.tvec = TfidfVectorizer().fit(x)
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
