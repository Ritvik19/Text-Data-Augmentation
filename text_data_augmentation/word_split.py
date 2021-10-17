import random

import nltk
from tqdm.auto import tqdm


class WordSplit:
    def __init__(self, alpha=0.05, n_aug=4, seed=None, show_progress=True):
        self.alpha = alpha
        self.n_aug = n_aug
        self.seed = seed
        self.disable_progress = not show_progress

    def __split_word(self, word):
        chars = list(word)
        chars.insert(random.randint(1, len(chars) - 1), " ")
        return "".join(word)

    def __word_split_aug(self, sentence):
        words = nltk.word_tokenize(sentence)
        words = [
            word if random.random() < self.alpha else self.__split_word(word)
            for word in words
        ]
        return " ".join(words)

    def __call__(self, x):
        random.seed(self.seed)
        augmented = []
        for sentence in tqdm(x, disable=self.disable_progress):
            augmented.append(sentence)
            for _ in range(self.n_aug):
                augmented.append(self.__word_split_aug(sentence))
        return augmented
