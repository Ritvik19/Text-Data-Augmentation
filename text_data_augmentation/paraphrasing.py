import random

import torch
from tqdm.auto import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


class Paraphrase:
    def __init__(self, t5model, n_aug=10, show_progress=True):
        self.tokenizer = T5Tokenizer.from_pretrained(t5model)
        self.model = T5ForConditionalGeneration.from_pretrained(t5model)
        self.n_aug = n_aug
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.disable_progress = not show_progress

    def __paraphrase(self, sentence):
        text = "paraphrase: " + sentence
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

    def __call__(self, x):
        augmented = []
        for sentence in tqdm(x, disable=self.disable_progress):
            augmented.append(sentence)
            augmented.extend(self.__paraphrase(sentence))
        return augmented
