import random

import nltk
from tqdm.auto import tqdm


class SynonymReplacement:
    def __init__(self, alpha=0.1, n_aug=4, seed=None, show_progress=True):
        self.alpha = alpha
        self.n_aug = n_aug
        self.seed = seed
        self.disable_progress = not show_progress
        self.stopwords = nltk.corpus.stopwords.words("english")

    def __get_synonym(self, word):
        synonyms = set()
        for syn in nltk.corpus.wordnet.synsets(word):
            for l in syn.lemmas():
                synonyms.add(l.name())
        synonyms = [word] if len(synonyms) == 0 else list(synonyms)
        return random.choice(synonyms)

    def __replace_word(self, words):
        r_idx = random.randint(0, len(words) - 1)
        while words[r_idx] in self.stopwords:
            r_idx = random.randint(0, len(words) - 1)
        syn = self.__get_synonym(words[r_idx])
        words[r_idx] = syn
        return words

    def __replace_sent(self, sentence):
        words = nltk.word_tokenize(sentence)
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