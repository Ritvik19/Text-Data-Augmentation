# Text-Data-Augmentation

State of the Text Data Augmentation for Natural Language Processing Applications

## Table of Contents

- [Text-Data-Augmentation](#text-data-augmentation)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Back Translation](#back-translation)
    - [Character Noise](#character-noise)
    - [Contextual Word Replacement](#contextual-word-replacement)
    - [Easy Data Augmentation](#easy-data-augmentation)
    - [KeyBoard Noise](#keyboard-noise)
    - [OCR Noise](#ocr-noise)
    - [Paraphrase](#paraphrase)
    - [Similar Word Replacement](#similar-word-replacement)
    - [Synonym Replacement](#synonym-replacement)
    - [Word Split](#word-split)
  - [References](#references)

---

## Installation

```bash
pip install git+https://github.com/Ritvik19/Text-Data-Augmentation.git
```

---

## Usage

This library various techniques for augmenting text data:

### Back Translation

Back Translation Augmentation relies on translating text data to another language and then translating it back to the original language. This technique allows generating textual data of distinct wording to original text while preserving the original context and meaning.[[1]](#ref-1) [[2]](#ref-2)

```python
>>> from text_data_augmentation import BackTranslation
>>> aug = BackTranslation()
>>> aug(['A quick brown fox jumps over the lazy dog'])
['A quick brown fox jumps over the lazy dog', 'A quick brown fox jumps on the lazy dog']
```

### Character Noise

Character Noise Augmentation adds character level noise by randomly inserting, deleting, swaping or replacing some charaters in the input text. [[2]](#ref-2)

```python
>>> from text_data_augmentation import CharacterNoise
>>> aug = CharacterNoise(alpha=0.1, n_aug=1)
>>> aug(['A quick brown fox jumps over the lazy dog'])
['A quick brown fox jumps over the lazy dog', 'A quick brown fox jumps ovr the lazy dog']
```

### Contextual Word Replacement

Contextual Word Replacement Augmentation creates Augmented Samples by randomly replacing some words with a mask and then using a Masked Language Model to fill it. [[2]](#ref-2) [[3]](#ref-3)

```python
>>> from text_data_augmentation import ContextualWordReplacement
>>> aug = ContextualWordReplacement(n_aug=1)
>>> aug(['A quick brown fox jumps over the lazy dog'])
['A quick brown fox jumps over the lazy dog', 'A quick brown fox jumps over his lazy dog']
```

### Easy Data Augmentation

Easy Data Augmentation adds word level noise by randomly inserting, deleting, swaping some words in the input text or by shuffling the sentences in the input text. [[4]](#ref-4) [[5]](#ref-5)

```python
>>> from text_data_augmentation import EasyDataAugmentation
>>> aug = EasyDataAugmentation(n_aug=1)
>>> aug(['A quick brown fox jumps over the lazy dog'])
['A quick brown fox jumps over the lazy dog', 'A quick brown fox jumps over the dog']
```

### KeyBoard Noise

KeyBoard Noise Augmentation adds character level spelling mistake noise by mimicing typographical errors made using a qwerty keyboard in the input text. [[2]](#ref-2)

```python
>>> from text_data_augmentation import KeyBoardNoise
>>> aug = KeyBoardNoise(alpha=0.1, n_aug=1)
>>> aug(['A quick brown fox jumps over the lazy dog'])
['A quick brown fox jumps over the lazy dog', 'A quick broen fox jumps over the lazy dog']
```

### OCR Noise

OCR Noise Augmentation adds character level spelling mistake noise by mimicing ocr errors in the input text. [[6]](#ref-6)

```python
>>> from text_data_augmentation import OCRNoise
>>> aug = OCRNoise(alpha=0.1, n_aug=1)
>>> aug(['A quick brown fox jumps over the lazy dog'])
['A quick brown fox jumps over the lazy dog', 'A quick hrown lox jumps over the lazy dog']
```

### Paraphrase

Paraphrase Augmentation rephrases the input sentences using T5 models. [[2]](#ref-2)

```python
>>> from text_data_augmentation import Paraphrase
>>> aug = Paraphrase("<T5 Model>", n_aug=1)
>>> aug(['A quick brown fox jumps over the lazy dog'])
['A quick brown fox jumps over the lazy dog', 'A quick brown fox has jumped on the lazy dog.']
```

### Similar Word Replacement

Similar Word Replacement Augmentation creates Augmented Samples by randomly replacing some words with a word having the most similar vector to it. [[2]](#ref-2) [[7]](#ref-7)

```python
>>> from text_data_augmentation import SimilarWordReplacement
>>> aug = SimilarWordReplacement("en_core_web_lg",  alpha=0.1, n_aug=1)
>>> aug(['A quick brown fox jumps over the lazy dog'])
['A quick brown fox jumps over the lazy dog', 'A quick White Wolf jumps over the lazy Cat.']
```

### Synonym Replacement

Synonym Replacement Augmentation creates Augmented Samples by randomly replacing some words with their synonyms based on the word net data base.[[2]](#ref-2) [[4]](#ref-4) [[8]](#ref-8)

```python
>>> from text_data_augmentation import SynonymReplacement
>>> aug = SynonymReplacement(alpha=0.1, n_aug=1)
>>> aug(['A quick brown fox jumps over the lazy dog'])
['A quick brown fox jumps over the lazy dog', 'A quick brown fox jumps over the lethargic dog']
```

### Word Split

Word Split Augmentation adds word level spelling mistake noise by spliting words randomly in the input text. [[2]](#ref-2)

```python
>>> from text_data_augmentation import WordSplit
>>> aug = WordSplit(alpha=0.1, n_aug=1)
>>> aug(['A quick brown fox jumps over the lazy dog'])
['A quick brown fox jumps over the lazy dog', 'A quick brown fox jumps over th e lazy dog']
```

---

## References

1. <a href="https://arxiv.org/pdf/2106.04681.pdf" id="ref-1">Data Expansion Using Back Translation and Paraphrasing for Hate Speech Detection</a>
2. <a href="https://arxiv.org/ftp/arxiv/papers/2107/2107.03158.pdf" id="ref-2">A Survey on Data Augmentation for Text Classification</a>
3. <a href="https://arxiv.org/pdf/1805.06201.pdf" id="ref-3">Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations</a>
4. <a href="https://arxiv.org/pdf/1901.11196.pdf" id="ref-4">EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks</a>
5. <a href="https://aclanthology.org/2020.coling-main.343.pdf" id="ref-5">An Analysis of Simple Data Augmentation for Named Entity Recognition</a>
6. <a href="https://zenodo.org/record/3245169/files/JCDL2019_Deep_Analysis.pdf" id="ref-6">Deep Statistical Analysis of OCR Errors for Effective Post-OCR Processing</a>
7. <a href="https://www.researchgate.net/publication/331784439_A_Study_of_Various_Text_Augmentation_Techniques_for_Relation_Classification_in_Free_Text" id="ref-7">A Study of Various Text Augmentation Techniques for Relation Classification in Free Text</a>
8. <a href="http://ceur-ws.org/Vol-2268/paper11.pdf" id="ref-8">Text Augmentation for Neural Networks</a>
