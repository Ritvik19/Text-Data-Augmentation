import random

import torch
from nltk import sent_tokenize
from tqdm.auto import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


class Paraphrase:
    """Paraphrase Augmentation rephrases the input sentences using T5 models.

    Args:
        t5model (string): T5 model
        n_aug (int, optional): Number of augmentations to be created for one sentence.
            Defaults to 10.
        show_progress (bool, optional): Set True to display progress bar.
            Defaults to True.
    """

    def __init__(self, t5model, n_aug=10, show_progress=True):
        self.tokenizer = T5Tokenizer.from_pretrained(t5model)
        self.model = T5ForConditionalGeneration.from_pretrained(t5model)
        self.n_aug = n_aug
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.disable_progress = not show_progress

    def __paraphrase_sent(self, sent):
        text = "paraphrase: " + sent
        encoding = self.tokenizer.encode_plus(
            text, pad_to_max_length=True, return_tensors="pt"
        )
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding[
            "attention_mask"
        ].to(self.device)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=self.n_aug,
        )
        return [
            self.tokenizer.decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for output in outputs
        ]

    def __paraphrase(self, sentence):
        sent_tokens = sent_tokenize(sentence)
        paraphrases = [self.__paraphrase_sent(sent) for sent in sent_tokens]
        paraphrases = [" ".join(list(x)) for x in zip(*paraphrases)]
        return paraphrases

    def __call__(self, x):
        augmented = []
        for sentence in tqdm(x, disable=self.disable_progress):
            augmented.extend(self.__paraphrase(sentence))
        return list(x) + augmented
