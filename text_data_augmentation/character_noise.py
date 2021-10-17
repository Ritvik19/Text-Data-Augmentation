import random

from tqdm.auto import tqdm


class CharacterNoise:
    def __init__(self, alpha=0.05, n_aug=4, seed=None, show_progress=True):
        self.alpha = alpha
        self.operations = ["insertion", "deletion", "swap", "replace"]
        self.n_aug = n_aug
        self.seed = seed
        self.disable_progress = not show_progress
        self.replacement_chars = {
            "a": list("qwsz"),
            "b": list("vghn"),
            "c": list("xdfv"),
            "d": list("serfcx"),
            "e": list("wsdr"),
            "f": list("drtgvc"),
            "g": list("ftyhbv"),
            "h": list("gyujnb"),
            "i": list("ujko"),
            "j": list("huikmn"),
            "k": list("jiolm"),
            "l": list("pok"),
            "m": list("njk"),
            "n": list("bhjm"),
            "o": list("iklp"),
            "p": list("ol"),
            "q": list("wsa"),
            "r": list("edft"),
            "s": list("wedxza"),
            "t": list("rfgy"),
            "u": list("yhji"),
            "v": list("cfgb"),
            "w": list("qase"),
            "x": list("zsdc"),
            "y": list("tghu"),
            "z": list("asx"),
        }

    def __insert(self, words):
        r_idx_w = random.randint(0, len(words) - 1)
        word = list(words[r_idx_w])
        r_idx_i = random.randint(0, len(word) - 1)
        word.insert(r_idx_i, random.choice(self.replacement_chars[word[r_idx_i]]))
        word = "".join(word)
        words[r_idx_w] = word
        return words

    def __insertion(self, sentence):
        words = sentence.split()
        for _ in range(int(self.alpha * len(words))):
            words = self.__insert(words)
        return " ".join(words)

    def __deletion(self, sentence):
        chars = list(sentence)
        new_chars = [char for char in chars if random.random() > self.alpha]
        return "".join(new_chars)

    def __swap_chars(self, chars):
        r_idx_1 = random.randint(0, len(chars) - 1)
        r_idx_2 = random.randint(0, len(chars) - 1)
        chars[r_idx_1], chars[r_idx_2] = chars[r_idx_2], chars[r_idx_1]
        return chars

    def __swap(self, sentence):
        chars = list(sentence)
        for _ in range(int(self.alpha * len(chars))):
            chars = self.__swap_chars(chars)
        return "".join(chars)

    def __replace(self, sentence):
        chars = list(sentence)
        new_chars = [
            char
            if random.random() > self.alpha
            else self.replacement_chars.get(char, char)
            for char in chars
        ]
        return "".join(new_chars)

    def __call__(self, x):
        random.seed(self.seed)
        augmented = []
        for sentence in tqdm(x, disable=self.disable_progress):
            augmented.append(sentence)
            for _ in range(self.n_aug):
                operation = random.choice(self.operations)
                if operation == "insertion":
                    augmented.append(self.__insertion(sentence))
                elif operation == "deletion":
                    augmented.append(self.__deletion(sentence))
                elif operation == "swap":
                    augmented.append(self.__swap(sentence))
                elif operation == "replace":
                    augmented.append(self.__replace(sentence))
                else:
                    raise AttributeError(
                        f"Invalid operation {operation}, valid operations are:"
                        + "insertion, deletion, swap, replace."
                    )
        return augmented
