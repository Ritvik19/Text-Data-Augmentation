import random
import re

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
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
        use_tfidf (bool, optional): Whether to use TFIDF weights to sample a word in the
            sentence. Defaults to True.
        seed (int, optional): Random State for reproducibility. Defaults to None.
        show_progress (bool, optional): Set True to display progress bar.
            Defaults to True.
    """

    def __init__(
        self, model, alpha=0.1, n_aug=4, use_tfidf=True, seed=None, show_progress=True
    ):
        self.alpha = alpha
        self.n_aug = n_aug
        self.use_tfidf = use_tfidf
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
        if self.use_tfidf:
            sentence = " ".join(words)
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
            syn = self.__get_similar_word(word_2_replace)
            return re.sub(word_2_replace, syn, sentence, 1, re.IGNORECASE)
        else:
            r_idx = random.randint(0, len(words) - 1)
            syn = self.__get_similar_word(words[r_idx])
            words[r_idx] = syn
            return " ".join(words)

    def __replace_sent(self, sentence):
        words = sentence.split()
        for _ in range(int(self.alpha * len(words))):
            aug_sent = self.__replace_word(words)
        return aug_sent

    def __call__(self, x):
        random.seed(self.seed)
        if self.use_tfidf:
            self.tvec = TfidfVectorizer().fit(x)
        augmented = []
        for sentence in tqdm(x, disable=self.disable_progress):
            for _ in range(self.n_aug):
                augmented.append(self.__replace_sent(sentence))
        return list(x) + augmented
