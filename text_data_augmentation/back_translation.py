from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, MarianTokenizer
from transformers.pipelines import base


class BackTranslation:
    """Back translation augmentation relies on translating text data to
    another language and then translating it back to the original language.
    This technique allows generating textual data of distinct wording to
    original text while preserving the original context and meaning.

    Args:
        base_language (str, optional): Base language of the text. Defaults to "en".
        interim_language (str, optional): Intermediate language to translatethe text to.
            Defaults to "fr".
        show_progress (bool, optional): Set True to display progress bar.
            Defaults to True.
    """

    def __init__(self, base_language="en", interim_language="fr", show_progress=True):
        self.disable_progress = not show_progress
        self.tokenizer_a = MarianTokenizer.from_pretrained(
            f"Helsinki-NLP/opus-mt-{base_language}-{interim_language}"
        )
        self.model_a = AutoModelForSeq2SeqLM.from_pretrained(
            f"Helsinki-NLP/opus-mt-{base_language}-{interim_language}"
        )
        self.tokenizer_b = MarianTokenizer.from_pretrained(
            f"Helsinki-NLP/opus-mt-{interim_language}-{base_language}"
        )
        self.model_b = AutoModelForSeq2SeqLM.from_pretrained(
            f"Helsinki-NLP/opus-mt-{interim_language}-{base_language}"
        )

    def __call__(self, x):
        augmented = []
        for doc in tqdm(x, disable=self.disable_progress):
            augmented.append(doc)
            try:
                input_ids_a = self.tokenizer_a.encode(doc, return_tensors="pt")
                outputs_a = self.model_a.generate(input_ids_a)
                decoded_a = self.tokenizer_a.decode(
                    outputs_a[0], skip_special_tokens=True
                )
                input_ids_b = self.tokenizer_b.encode(decoded_a, return_tensors="pt")
                outputs_b = self.model_b.generate(input_ids_b)
                decoded_b = self.tokenizer_b.decode(
                    outputs_b[0], skip_special_tokens=True
                )
                augmented.append(decoded_b)
            except IndexError:
                augmented.append(doc)

        return augmented
