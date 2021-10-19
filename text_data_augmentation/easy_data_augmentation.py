import random

import nltk
from tqdm.auto import tqdm


class EasyDataAugmentation:
    """Easy Data Augmentation adds word level noise by randomly inserting,
    deleting, swaping some words in the input text or by shuffling the
    sentences in the input text.

    Args:
        alpha (float, optional): Control parameter, frequency of operation increases
            with increase in the vvalue of alpha. Defaults to 0.01.
        n_aug (int, optional): Number of augmentations to be created for one sentence.
            Defaults to 4.
        operations (list, optional): List of operations to perform.
        Defaults to ["insertion", "deletion", "swap", "shuffle"].
        seed (int, optional): Random State for reproducibility. Defaults to None.
        show_progress (bool, optional): Set True to display progress bar.
            Defaults to True.
    """

    def __init__(
        self, alpha=0.1, n_aug=4, operations=None, seed=None, show_progress=True
    ):
        self.alpha = alpha
        self.operations = operations or ["insertion", "deletion", "swap", "shuffle"]
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

    def __insert(self, words):
        r_idx_w = random.randint(0, len(words) - 1)
        while words[r_idx_w] in self.stopwords:
            r_idx_w = random.randint(0, len(words) - 1)
        r_idx_i = random.randint(0, len(words) - 1)
        syn = self.__get_synonym(words[r_idx_w])
        words.insert(r_idx_i, syn)
        return words

    def __insertion(self, sentence):
        words = nltk.word_tokenize(sentence)
        for _ in range(int(self.alpha * len(words))):
            words = self.__insert(words)
        return " ".join(words)

    def __deletion(self, sentence):
        words = nltk.word_tokenize(sentence)
        new_words = [word for word in words if random.random() > self.alpha]
        return " ".join(new_words)

    def __swap_words(self, words):
        r_idx_1 = random.randint(0, len(words) - 1)
        r_idx_2 = random.randint(0, len(words) - 1)
        words[r_idx_1], words[r_idx_2] = words[r_idx_2], words[r_idx_1]
        return words

    def __swap(self, sentence):
        words = nltk.word_tokenize(sentence)
        for _ in range(int(self.alpha * len(words))):
            words = self.__swap_words(words)
        return " ".join(words)

    def __shuffle(self, doc):
        sentences = nltk.sent_tokenize(doc)
        random.shuffle(sentences)
        return " ".join(sentences)

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
                elif operation == "shuffle":
                    augmented.append(self.__shuffle(sentence))
                else:
                    raise AttributeError(
                        f"Invalid operation {operation}, valid operations are:"
                        + "insertion, deletion, swap, shuffle."
                    )
        return augmented
