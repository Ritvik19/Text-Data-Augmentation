import random

from tqdm.auto import tqdm


class CharacterNoise:
    """Character Noise Augmentation adds character level noise by randomly inserting,
    deleting, swaping or replacing some charaters in the input text.

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
        self.operations = ["insertion", "deletion", "swap", "replace"]
        self.n_aug = n_aug
        self.seed = seed
        self.disable_progress = not show_progress
        self.replacement_chars = list(
            "qwertyuiopasdfghjklzxcvbnm QWERTYUIOPASDFGHJKLZXCVBNM 1234567890"
        )

    def __insert(self, words):
        r_idx_w = random.randint(0, len(words) - 1)
        word = list(words[r_idx_w])
        r_idx_i = random.randint(0, len(word) - 1)
        word.insert(r_idx_i, random.choice(self.replacement_chars))
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
            else random.choice(self.replacement_chars)
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
