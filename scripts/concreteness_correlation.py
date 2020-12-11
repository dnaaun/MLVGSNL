## Imports
from __future__ import annotations
from collections import Counter
import pandas as pd
from itertools import islice
import stanza
from stanza.models.common.doc import Document, Word, Sentence
from more_itertools import split_before
import pandas as pd
import typer
from dnips.iter.bidict import Ordering
from dnips.iter import myzip
import tqdm
import numpy as np
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Sized,
    Tuple,
    Union,
    overload,
)
from word2word import Word2word
from pathlib import Path

##


def bert_join_subwords(subwords: Sequence[str]) -> str:
    new_subwords = []
    for subword in subwords:
        if subword[:2] == "##":
            assert len(subword) > 2
            subword = subword[2:]
        new_subwords.append(subword)
    return "".join(new_subwords)


def read_conc_scores(english_conc_file: Path, conc_score_col: str) -> pd.Series[float]:
    df = pd.read_csv(english_conc_file, sep="\t")
    print(f"Read {len(df)} columns from english concretenesss CSV.")
    df["Word"] = df["Word"].str.lower()
    df.set_index("Word", inplace=True)
    scores = df[conc_score_col]
    return scores


class StanzaLemmatizer:
    """The shenanigans below (namely, adding _MARKER tokens, joining multiple
    words into one by \n\n (which stanza recognizes as sentence sep) )
    is to enable batched prediction."""

    _MARKER = "BEGIN"
    _STANZA_DIR = "/projectnb/llamagrp/davidat/stanza_resources/"

    def __init__(self, pipeline: stanza.Pipeline) -> None:
        self._pline = pipeline

    @classmethod
    def by_lang(cls, lang: str) -> StanzaLemmatizer:
        pline = stanza.Pipeline(
            lang,
            processors="tokenize,lemma",
            tokenize_batch_size=8000,
            lemma_batch_size=8000,
            dir=cls._STANZA_DIR,
        )
        return cls(pipeline=pline)

    @overload
    def __call__(self, __words: str) -> Optional[str]:
        ...

    @overload
    def __call__(self, __words: List[str]) -> List[Optional[str]]:
        ...

    def __call__(
        self, input_: Union[str, List[str]]
    ) -> Union[Optional[str], List[Optional[str]]]:
        if isinstance(input_, str):
            return self._do_on_list([input_])[0]
        else:
            return self._do_on_list(input_)

    def _do_on_list(self, words: List[str]) -> List[Optional[str]]:
        words = self._add_marker_toks(words)
        doc = self._pline("\n\n".join(words))
        per_orig_word = list(
            split_before(doc.sentences, lambda sent: sent.words[0].text == self._MARKER)
        )
        res = list(map(self.extract_lemma, per_orig_word))
        return res

    def _add_marker_toks(self, batch: List[str]) -> List[str]:
        return [f"{self._MARKER} {s}" for s in batch]

    def extract_lemma(self, sents: List[Sentence]) -> Optional[str]:
        if len(sents) == 0 or len(sents[0].words) == 0:
            raise Exception("This should never happen.")
        if len(sents) != 1:  # word pasred as multiple sentences
            return None
        if len(sents[0].words) != 2:  # word parsed as mutliple words
            return None
        # sents[0].words[1] is BEGIN
        return sents[0].words[1].lemma


def get_word_concreteness(
    en_conc_scores: pd.Series[float], words: pd.Index[str], lang: str
) -> pd.Series[float]:

    # Setup lemmatization

    translator: Callable[[str], Optional[str]]
    en_lemmatizer: Callable[[str], Optional[str]]
    other_lemmatizer: Callable[[str], Optional[str]]

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

    other_failed_to_lemmatize = 0
    trans_failed_to_lemmatize = 0
    failed_to_trans = 0

    no_score_for_lemma = 0
    word_scores: List[Optional[float]] = []
    for word in tqdm.tqdm(words, desc=f"Getting word concreteness for {lang}"):
        # Lemmatize before translate
        lemma = other_lemmatizer(word)
        if lemma is None:
            word_scores.append(None)
            other_failed_to_lemmatize += 1
            continue

        translation = translator(lemma)

        if translation is None:
            word_scores.append(None)
            failed_to_trans += 1
            continue

        # Lemmatize after translate
        en_lemma: Optional[str] = en_lemmatizer(translation)  # type: ignore[misc]
        if en_lemma is None:
            word_scores.append(None)
            trans_failed_to_lemmatize += 1
            continue

        score = None
        if en_lemma in en_conc_scores.index:
            score = en_conc_scores[en_lemma]
        if score is None:
            no_score_for_lemma += 1
        word_scores.append(score)

    num_words = sum(0 if s is None else 1 for s in word_scores)
    print(
        f"For language {lang}, found concreteness measure "
        f"for {num_words} words, out of {len(words)}."
        f" {other_failed_to_lemmatize} words in '{lang}' failed to lemmatize."
        f" {failed_to_trans} lemmas failed to translate."
        f" {trans_failed_to_lemmatize} translations to english failed to lemmatize."
        f" {no_score_for_lemma} lemmas had no scores."
    )
    scores, words = myzip(  # type: ignore
        [(score, word) for score, word in zip(word_scores, words) if score is not None]
    )
    sr = pd.Series(scores, index=words)
    return sr


def get_word_subword_matrix(
    train_file: Path,
    subword_sep: str = "|",
    join_subwords_func: Callable[[Sequence[str]], str] = bert_join_subwords,
) -> pd.DataFrame:
    with train_file.open() as f:
        words_per_line = [line.strip().split() for line in f]

    subwords_per_word_per_line = [
        [word.split(subword_sep) for word in sent] for sent in words_per_line
    ]

    # Remove "subword marks". For eg, BERT adds ## to subwords that don't appear
    # at beginining of word
    words = {}
    for sent in subwords_per_word_per_line:
        for subwords in sent:
            word = join_subwords_func(subwords)
            subword_cnt = dict(Counter(subwords))
            if word not in words:
                words[word] = subword_cnt
            else:
                # check to make sure sure all subword tokenizations of this word are teh same
                assert words[word] == subword_cnt
    res = pd.DataFrame.from_dict(words).T.fillna(0)

    return res


def get_single_lang_stats(
    mat: np.ndarray, subword_dim: int, word_dim: int
) -> Dict[str, float]:

    return {
        "Average num of subwords in word": (mat > 0).sum(axis=subword_dim).mean(),
        "Average num of words that a subword appears in": (mat > 0)
        .sum(axis=word_dim)
        .mean(),
        "Num subwords": mat.shape[0],
        "Num words": mat.shape[1],
    }


##  End of helper funcs


if False:
## We run this manually through vim-ipy only
    lang1 = "en"
    lang2 = "de"
    lang1_train_file = Path("/project/llamagrp/davidat/mlvgsnl/concat_dset_data/multi30k_en/train_caps_subword.txt")
    lang2_train_file = Path("/project/llamagrp/davidat/mlvgsnl/concat_dset_data/multi30k_de/train_caps_subword.txt")
    concreteness_score_file = Path(
        "/projectnb/statnlp/davidat/mlvgsnl/Concreteness_English.txt"
    )
    conc_score_col = "Conc.M"
##


def main(
    lang1: str,
    lang2: str,
    lang1_train_file: Path,
    lang2_train_file: Path,
    concreteness_score_file: Path,
    conc_score_col: str = "Conc.M",
) -> None:
    lang1_word_subword_mat = get_word_subword_matrix(lang1_train_file)
    lang2_word_subword_mat = get_word_subword_matrix(lang2_train_file)

    for lang, df in [(lang1, lang1_word_subword_mat), (lang2, lang2_word_subword_mat)]:
        print(
            f"For lang {lang}",
            get_single_lang_stats(df.to_numpy(), word_dim=0, subword_dim=1),
        )

    en_conc_scores = read_conc_scores(
        english_conc_file=concreteness_score_file,
        conc_score_col=conc_score_col,
    )
    lang1_word_conc = get_word_concreteness(
        en_conc_scores=en_conc_scores,
        words=lang1_word_subword_mat.index,
        lang=lang1,
    )
    lang2_word_conc = get_word_concreteness(
        en_conc_scores=en_conc_scores,
        words=lang2_word_subword_mat.index,
        lang=lang2,
    )

    # Filter out words without concreteness scores
    lang1_word_subword_mat = lang1_word_subword_mat.reindex(
        lang1_word_subword_mat.index & lang1_word_conc.index
    )
    lang2_word_subword_mat = lang2_word_subword_mat.reindex(
        lang2_word_subword_mat.index & lang2_word_conc.index
    )
    print(
        f"After filtering out words not in concreteness lexicon, "
        f"{lang1} has {lang1_word_subword_mat.shape[0]} words, "
        f"{lang2} has {lang2_word_subword_mat.shape[0]} words. "
    )

    # Remove subwords that are not needed anymore
    lang1_still_there_subwords = lang1_word_subword_mat.sum(axis=0) > 0
    lang2_still_there_subwords = lang2_word_subword_mat.sum(axis=0) > 0

    lang1_word_subword_mat = lang1_word_subword_mat.loc[:, lang1_still_there_subwords]
    lang2_word_subword_mat = lang2_word_subword_mat.loc[:, lang2_still_there_subwords]

    print(
        f"After filtering out subwords that are not needed by concrete words, "
        f"{lang1} has {lang1_word_subword_mat.shape[1]} subwords, "
        f"{lang2} has {lang2_word_subword_mat.shape[1]} subwords. "
    )

    # Get common subwords
    comm_subwords = lang1_word_subword_mat.columns & lang2_word_subword_mat.columns
    num_all_subwords = len(
        lang1_word_subword_mat.columns | lang2_word_subword_mat.columns
    )
    print(
        f"Found {len(comm_subwords)} common subwords out of {num_all_subwords} subwords in total."
        " For {}, that is {:.1%}".format(
            lang1, len(comm_subwords) / lang1_word_subword_mat.shape[1]
        )
        + " For {}, that is {:.1%}".format(
            lang2, len(comm_subwords) / lang2_word_subword_mat.shape[1]
        )
    )
    # Narrow down to common subwords
    lang1_word_subword_mat = lang1_word_subword_mat.loc[:, comm_subwords]
    lang2_word_subword_mat = lang2_word_subword_mat.loc[:, comm_subwords]
    "bol" in comm_subwords

    # Narrow down to words that can be formed with the common subwords
    lang1_made_of_comm_subwords_filter = lang1_word_subword_mat.sum(axis=1) > 0
    lang2_made_of_comm_subwords_filter = lang2_word_subword_mat.sum(axis=1) > 0

    print(
        "After filtering out words that don't use subwords from other lang, we are left with num of words:\t"
        f"{lang1}:  {lang1_made_of_comm_subwords_filter.sum()}; "
        f"{lang2}:  {lang2_made_of_comm_subwords_filter.sum()}."
    )


    lang1_word_subword_mat = lang1_word_subword_mat.loc[lang1_made_of_comm_subwords_filter]
    lang2_word_subword_mat = lang2_word_subword_mat.loc[lang2_made_of_comm_subwords_filter]

    if lang1_word_subword_mat.shape[1] != lang2_word_subword_mat.shape[1]:
        raise Exception(
            f"word-subword matrices narrowed down to different num of subwords. Must be a bug."
        )

    # Compute subword concreteness
    # Let W_s be the set of all words with concreteness scores that subword s appears in
    # the subword concretenss is defined as the weighted average of the concreteness
    # scores of words in W_s, where the weight is the percentage of times the subword
    # appears in that word
    lang1_word_weights = lang1_word_subword_mat / lang1_word_subword_mat.to_numpy().sum(
        axis=0, keepdims=True
    )
    lang2_word_weights = lang2_word_subword_mat / lang2_word_subword_mat.to_numpy().sum(
        axis=0, keepdims=True
    )

    assert lang1_word_weights.shape == lang1_word_subword_mat.shape
    assert np.allclose(lang1_word_weights.sum(axis=0), 1)  # type: ignore[arg-type]

    lang1_subword_conc = (
        lang1_word_subword_mat * lang1_word_weights
    ).T @ lang1_word_conc.loc[lang1_word_subword_mat.index]
    lang2_subword_conc = (
        lang2_word_subword_mat * lang2_word_weights
    ).T @ lang2_word_conc.loc[lang2_word_subword_mat.index]

    corr = np.corrcoef(lang1_subword_conc, lang2_subword_conc)
    print(f"The subword concreteness pearson correlation is: {corr}")


if __name__ == "__main__":
    typer.run(main)
