import random

from tqdm.auto import tqdm
from transformers import pipeline


class ContextualWordReplacement:
    def __init__(
        self, model=None, n_aug=10, seed=None, show_progress=True, return_best=False
    ):
        self.model = pipeline("fill-mask", model=model)
        self.n_aug = n_aug
        self.seed = seed
        self.return_best = return_best
        self.disable_progress = not show_progress

    def __contextual_word_replacement(self, sentence):
        words = sentence.split()
        r_idx = random.randint(0, len(words) - 1)
        words[r_idx] = "<mask>"
        masked_sentence = " ".join(words)
        return self.model(masked_sentence)

    def __call__(self, x):
        random.seed(self.seed)
        augmented = []
        for sentence in tqdm(x, disable=self.disable_progress):
            for _ in range(self.n_aug):
                augmented.extend(self.__contextual_word_replacement(sentence))
        augmented = list(sorted(augmented, key=lambda s: s["score"], reverse=True))
        augmented = [s["sequence"] for s in augmented]
        if not self.return_best:
            random.shuffle(augmented)
        augmented = augmented[: self.n_aug]
        return x + augmented
