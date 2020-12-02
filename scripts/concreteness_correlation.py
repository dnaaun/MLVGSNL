from __future__ import annotations
from itertools import islice

import stanza  # type: ignore
import pandas as pd
import typer
from dnips.iter.bidict import Ordering
import tqdm
import numpy as np
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple
from word2word import Word2word
from pathlib import Path


def get_word_subword_matrix(
    subword_order: Ordering[str],
    word_order: Ordering[str],
    examples: List[List[List[str]]],
) -> "np.ndarray[np.float64]":
    mat = np.zeros((len(word_order), len(subword_order)), dtype=np.float64)

    for sent in examples:
        for word in sent:
            word_idx = word_order.indices["".join(word)]
            for subword in word:
                subword_idx = subword_order.indices[subword]
                mat[word_idx, subword_idx] += 1
    return mat


def read_conc_scores(english_conc_file: Path, conc_score_col: str) -> Dict[str, float]:
    df = pd.read_csv(english_conc_file, sep="\t")
    print(f"Read {len(df)} columns from english concretenesss CSV.")
    lemmas = df["Word"].str.lower()
    scores = df[conc_score_col].tolist()
    conc_scores = {lemma: score for lemma, score in zip(lemmas, scores)}
    return conc_scores


class StanzaLemmatizer:
    def __init__(self, pipeline: stanza.Pipeline) -> None:
        self._pline = pipeline

    @classmethod
    def by_lang(cls, lang: str) -> StanzaLemmatizer:
        pline = stanza.Pipeline(lang, processors="tokenize,lemma")
        return cls(pipeline=pline)

    def __call__(self, word: str) -> Optional[str]:
        doc = self._pline(word)
        if len(doc.sentences) != 1:
            return None
        if len(doc.sentences[0].words) != 1:  # More than one token
            return None
        return doc.sentences[0].words[0].lemma


def get_word_concreteness(
    en_conc_scores: Dict[str, float], words: Ordering, lang: str
) -> List[Optional[float]]:

    # Setup lemmatization

    en_lemmatizer = StanzaLemmatizer.by_lang("en")
    if lang == "en":
        other_lemmatizer = lambda x: x
        translator = lambda x: x
    else:
        other_lemmatizer = StanzaLemmatizer.by_lang(lang)

        dict_ = Word2word(lang, "en")

        def translator(word: str) -> Optional[str]:
            try:
                return dict_(word)[0].lower()
            except KeyError:
                return None

    word_scores: List[Optional[float]] = []
    for word in tqdm.tqdm(words, desc=f"Getting word concreteness for {lang}"):
        # Lemmatize before translate
        lemma = other_lemmatizer(word)
        if lemma is None:
            word_scores.append(None)
            continue

        translation = translator(lemma)

        if translation is None:
            word_scores.append(None)
            continue

        # Lemmatize after translate
        en_lemma = en_lemmatizer(translation)
        if en_lemma is None:
            word_scores.append(None)
            continue

        score = en_conc_scores.get(en_lemma, None)
        word_scores.append(score)

    num_words = sum(0 if s is None else 1 for s in word_scores)
    print(
        f"For language {lang}, found concreteness measure "
        f"for {num_words} words, out of {len(words)}."
    )
    return word_scores


def get_subword_examples(
    train_file: Path, subword_sep: str = "|"
) -> Tuple[Ordering[str], Ordering[str], List[List[List[str]]]]:
    with train_file.open() as f:
        words_per_line = [line.strip().split() for line in f]

    subwords_per_word_per_line = [
        [word.split(subword_sep) for word in sent] for sent in words_per_line
    ]

    # Remove "subword marks". For eg, BERT adds ## to subwords that don't appear
    # at beginining of word
    final = []
    unique_subwords = set()
    unique_words = set()
    for sent in subwords_per_word_per_line:
        new_sent = []
        for word in sent:
            new_word = [word[0]]

            for subword in word[1:]:
                if subword[:2] == "##":
                    assert len(subword) > 2
                    new_subword = subword[2:]
                else:
                    new_subword = subword

                new_word.append(new_subword)
                unique_subwords.add(new_subword)

            unique_words.add("".join(new_word))
            new_sent.append(new_word)
        final.append(new_sent)
    return Ordering(sorted(unique_subwords)), Ordering(sorted(unique_words)), final


def main(
    lang1: str,
    lang2: str,
    lang1_train_file: Path,
    lang2_train_file: Path,
    concreteness_score_file: Path,
    conc_score_col: str = "Conc.M",
) -> None:
    lang1_subword_order, lang1_word_order, lang1_exs = get_subword_examples(
        lang1_train_file
    )[:5]
    lang2_subword_order, lang2_word_order, lang2_exs = get_subword_examples(
        lang2_train_file
    )[:5]
    en_conc_scores = read_conc_scores(
        english_conc_file=concreteness_score_file,
        conc_score_col=conc_score_col,
    )
    # lang2_exs = get_subword_examples(lang2_train_file)[:5]
    lang1_word_conc = get_word_concreteness(
        en_conc_scores=en_conc_scores,
        words=lang1_word_order[-50:],
        lang=lang1,
    )
    lang2_word_conc = get_word_concreteness(
        en_conc_scores=en_conc_scores,
        words=lang2_word_order[-50:],
        lang=lang2,
    )

    lang1_word_subword_mat = get_word_subword_matrix(
        lang1_subword_order, lang1_word_order, lang1_exs
    )
    lang2_word_subword_mat = get_word_subword_matrix(
        lang2_subword_order, lang2_word_order, lang2_exs
    )
    breakpoint()


if __name__ == "__main__":
    typer.run(main)
