import random

import nltk
from tqdm.auto import tqdm


class WordSplit:
    """Word Split Augmentation adds word level spelling mistake noise by spliting
    words randomly in the input text.

    Args:
        alpha (float, optional): Control parameter, frequency of operation increases
            with increase in the vvalue of alpha. Defaults to 0.01.
        n_aug (int, optional): Number of augmentations to be created for one sentence.
            Defaults to 4.
        seed (int, optional): Random State for reproducibility. Defaults to None.
        show_progress (bool, optional): Set True to display progress bar.
            Defaults to True.
    """

    def __init__(self, alpha=0.1, n_aug=4, seed=None, show_progress=True):
        self.alpha = alpha
        self.n_aug = n_aug
        self.seed = seed
        self.disable_progress = not show_progress

    def __split_word(self, word):
        chars = list(word)
        if len(chars) > 2:
            chars.insert(random.randint(1, len(chars) - 1), " ")
        return "".join(chars)

    def __word_split_aug(self, sentence):
        words = nltk.word_tokenize(sentence)
        words = [
            word if random.random() > self.alpha else self.__split_word(word)
            for word in words
        ]
        return " ".join(words)

    def __call__(self, x):
        random.seed(self.seed)
        augmented = []
        for sentence in tqdm(x, disable=self.disable_progress):
            for _ in range(self.n_aug):
                augmented.append(self.__word_split_aug(sentence))
        return list(x) + augmented
