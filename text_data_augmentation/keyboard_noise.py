import random

from tqdm.auto import tqdm


class KeyBoardNoise:
    """KeyBoard Noise Augmentation adds character level spelling mistake noise by
    mimicing typographical errors made using a qwerty keyboard in the input text.

    Args:
        alpha (float, optional): Control parameter, frequency of operation increases
            with increase in the vvalue of alpha. Defaults to 0.01.
        n_aug (int, optional): Number of augmentations to be created for one sentence.
            Defaults to 4.
        seed (int, optional): Random State for reproducibility. Defaults to None.
        show_progress (bool, optional): Set True to display progress bar.
            Defaults to True.
    """

    def __init__(self, alpha=0.01, n_aug=4, seed=None, show_progress=True):
        self.alpha = alpha
        self.n_aug = n_aug
        self.seed = seed
        self.disable_progress = not show_progress
        self.replacement_chars = {
            "1": ["!", "2", "@", "q", "w"],
            "2": ["@", "1", "!", "3", "#", "q", "w", "e"],
            "3": ["#", "2", "@", "4", "$", "w", "e"],
            "4": ["$", "3", "#", "5", "%", "e", "r"],
            "5": ["%", "4", "$", "6", "^", "r", "t", "y"],
            "6": ["^", "5", "%", "7", "&", "t", "y", "u"],
            "7": ["&", "6", "^", "8", "*", "y", "u", "i"],
            "8": ["*", "7", "&", "9", "(", "u", "i", "o"],
            "9": ["(", "8", "*", "0", ")", "i", "o", "p"],
            "!": ["@", "q"],
            "@": ["!", "#", "q", "w"],
            "#": ["@", "$", "w", "e"],
            "$": ["#", "%", "e", "r"],
            "%": "$",
            "q": ["1", "!", "2", "@", "w", "a", "s"],
            "w": ["1", "!", "2", "@", "3", "#", "q", "e", "a", "s", "d"],
            "e": ["2", "@", "3", "#", "4", "$", "w", "r", "s", "d", "f"],
            "r": ["3", "#", "4", "$", "5", "%", "e", "t", "d", "f", "g"],
            "t": ["4", "$", "5", "%", "6", "^", "r", "y", "f", "g", "h"],
            "y": ["5", "%", "6", "^", "7", "&", "t", "u", "g", "h", "j"],
            "u": ["6", "^", "7", "&", "8", "*", " t", "i", "h", "j", "k"],
            "i": ["7", "&", "8", "*", "9", "(", "u", "o", "j", "k", "l"],
            "o": ["8", "*", "9", "(", "0", ")", "i", "p", "k", "l"],
            "p": ["9", "(", "0", ")", "o", "l"],
            "a": ["q", "w", "a", "s", "z", "x"],
            "s": ["q", "w", "e", "a", "d", "z", "x", "c"],
            "d": ["w", "e", "r", "s", "f", "x", "c", "v"],
            "f": ["e", "r", "t", "d", "g", "c", "v", "b"],
            "g": ["r", "t", "y", "f", "h", "v", "b", "n"],
            "h": ["t", "y", "u", "g", "j", "b", "n", "m"],
            "j": ["y", "u", "i", "h", "k", "n", "m", ",", "<"],
            "k": ["u", "i", "o", "j", "l", "m", ",", "<", ".", ">"],
            "l": ["i", "o", "p", "k", ";", ":", ",", "<", ".", ">", "/", "?"],
            "z": ["a", "s", "x"],
            "x": ["a", "s", "d", "z", "c"],
            "c": ["s", "d", "f", "x", "v"],
            "v": ["d", "f", "g", "c", "b"],
            "b": ["f", "g", "h", "v", "n"],
            "n": ["g", "h", "j", "b", "m"],
            "m": ["h", "j", "k", "n", ",", "<"],
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
            for _ in range(self.n_aug):
                augmented.append(self.__replace(sentence))
        return list(x) + augmented
