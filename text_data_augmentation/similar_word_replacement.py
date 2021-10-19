import random

import numpy as np
import spacy
from tqdm.auto import tqdm


class SimilarWordReplacement:
    """Similar Word Replacement Augmentation creates Augmented Samples by randomly
    replacing some words with a word having the most similar vector to it.

    Args:
        model (string): Spacy Model to generate vectors.
        alpha (float, optional): Control parameter, frequency of operation increases
            with increase in the vvalue of alpha. Defaults to 0.01.
        n_aug (int, optional): Number of augmentations to be created for one sentence.
            Defaults to 4.
        seed (int, optional): Random State for reproducibility. Defaults to None.
        show_progress (bool, optional): Set True to display progress bar.
            Defaults to True.
    """

    def __init__(self, model, alpha=0.1, n_aug=4, seed=None, show_progress=True):
        self.alpha = alpha
        self.n_aug = n_aug
        self.seed = seed
        self.disable_progress = not show_progress
        self.nlp = spacy.load(model)

    def __get_similar_word(self, word):
        try:
            ms = self.nlp.vocab.vectors.most_similar(
                np.asarray([self.nlp.vocab.vectors[self.nlp.vocab.strings[word]]]), n=15
            )
            words = [self.nlp.vocab.strings[w] for w in ms[0][0]]
            words = [w for w in words if w.lower() != word.lower()]
            return random.choice(words)
        except (KeyError, IndexError):
            return word

    def __replace_word(self, words):
        r_idx = random.randint(0, len(words) - 1)
        syn = self.__get_similar_word(words[r_idx])
        words[r_idx] = syn
        return words

    def __replace_sent(self, sentence):
        words = sentence.split()
        for _ in range(int(self.alpha * len(words))):
            words = self.__replace_word(words)
        return " ".join(words)

    def __call__(self, x):
        random.seed(self.seed)
        augmented = []
        for sentence in tqdm(x, disable=self.disable_progress):
            augmented.append(sentence)
            for _ in range(self.n_aug):
                augmented.append(self.__replace_sent(sentence))
        return augmented
