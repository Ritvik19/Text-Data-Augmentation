from googletrans import Translator
from tqdm.auto import tqdm


class BackTranslation:
    def __init__(self, base_language, interim_language, show_progress=True):
        self.base_language = base_language
        self.interim_language = interim_language
        self.disable_progress = not show_progress
        self.translator = Translator()

    def __call__(self, x):
        augmented = []
        for doc in tqdm(x, disable=self.disable_progress):
            augmented.append(doc)
            try:
                aug = self.translator.translate(
                    text=doc, dest=self.interim_language
                ).text
                aug = self.translator.translate(text=aug, dest=self.base_language).text
                augmented.append(aug)
            except AttributeError:
                pass

        return augmented
