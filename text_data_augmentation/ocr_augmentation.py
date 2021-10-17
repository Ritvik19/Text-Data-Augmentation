import random

from tqdm.auto import tqdm


class OCRAugmentation:
    def __init__(self, alpha=0.05, n_aug=4, seed=None, show_progress=True):
        self.alpha = alpha
        self.n_aug = n_aug
        self.seed = seed
        self.disable_progress = not show_progress
        self.replacement_chars = {
            "a": ["@", "d", "u"],
            "b": ["h"],
            "c": ["e", "o"],
            "d": ["a", "@"],
            "e": ["o", "c"],
            "f": ["t", "l", "i", "j", "1"],
            "g": ["p", "q"],
            "h": ["b"],
            "i": ["l", "j", "t", "1"],
            "j": ["i", "l", "t", "1"],
            "k": ["k"],
            "l": ["i", "j", "t", "f", "1"],
            "m": ["m", "rn"],
            "n": ["n", "r"],
            "o": ["c", "e"],
            "p": ["q", "g"],
            "q": ["p", "g"],
            "r": ["r"],
            "s": ["5"],
            "t": ["f", "i", "j", "l", "1"],
            "u": ["a", "v"],
            "v": ["a", "u"],
            "w": ["vv", "uv", "vu", "vv"],
            "x": ["y"],
            "y": ["x"],
            "z": ["2"],
            "1": ["4", "7", "l", "i", "t", "f", "j"],
            "2": ["3", "z"],
            "3": ["2"],
            "4": ["1", "7"],
            "5": ["6", "s"],
            "6": ["5"],
            "7": ["1", "4", "l", "i", "t", "f", "j"],
            "8": ["9", "0", "o"],
            "9": ["0", "8", "o"],
            "0": ["8", "9", "o"],
        }

    def __replace(self, sentence):
        chars = list(sentence)
        new_chars = [
            char
            if random.random() > self.alpha
            else random.choice(self.replacement_chars.get(char, [char]))
            for char in chars
        ]
        return "".join(new_chars)

    def __call__(self, x):
        random.seed(self.seed)
        augmented = []
        for sentence in tqdm(x, disable=self.disable_progress):
            augmented.append(sentence)
            for _ in range(self.n_aug):
                augmented.append(self.__replace(sentence))
        return augmented
