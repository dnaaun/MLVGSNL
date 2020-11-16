#!/usr/bin/env python
import gc
from more_itertools import chunked
import pickle as pkl
from functools import reduce
from itertools import chain, islice
from pathlib import Path
import re
from io import TextIOWrapper
import typer
import tqdm
from typing import (
    Generic,
    TypeVar,
    TYPE_CHECKING,
    Tuple,
    Union,
    List,
    Any,
    Counter,
    Set,
    TextIO,
    Optional,
)
import zipfile, gzip

import lazy_import

if TYPE_CHECKING:  # These two modules take a long time so do lazy importing.
    import nltk  # type: ignore
    import numpy as np
    import transformers as ts
    import torch
    import benepar  # type: ignore

    from src.vocab import Vocabulary
else:
    nltk = lazy_import.lazy_module("nltk")
    np = lazy_import.lazy_module("numpy")
    ts = lazy_import.lazy_module("transformers")
    torch = lazy_import.lazy_module("torch")
    benepar = lazy_import.lazy_module("benepar")

    # Add the path of the mlvgnsl code so we can import Vocabulary.
    import sys

    src_path = str(Path(__file__).resolve().parent.parent / "src")
    sys.path.append(src_path)

    from vocab import Vocabulary


app = typer.Typer()

SPECIAL_TOKENS = [
    "<pad>",  # MUST BE FIRST, the codebase assumes vocab  id 0 is padding.
    "<unk>",
    "<start>",
    "<end>",
]


@app.command()
def convert_to_subword(
    txt_file: Path,
    output_file: Path,
    transformers_mdl: str = "bert-base-multilingual-uncased",
    subword_sep: str = "|",
) -> None:
    """

    Output the subword tokenization for a text file.
    NOTE: This uses bert-base-uncased, and so everything is lowercased.

    Args:
        txt_file: A text file, with one sentence per line. Like this:

            A restaurant has modern wooden tables and chairs .
            A long restaurant table with rattan rounded back chairs .
            a long table with a plant on top of it surrounded with wooden chairs
            A long table with a flower arrangement in the middle for meetings

        output_file: THe output file will contain the subword tokenization. Like this:

            a restaurant has modern wooden tables and chair|##s .
            a long restaurant table with ratt|##an rounded back chair|##s .
            a long table with a plant on top of it surrounded with wooden chair|##s
            a long table with a flower arrangement in the middle for meetings

        subword_sep: The subword separator used in the output file.
    """

    if len(subword_sep) != 1:
        raise Exception(f"Subword sep must be of length 1, not '{subword_sep}'")

    found_subwords = set()

    tokenizer = ts.AutoTokenizer.from_pretrained(transformers_mdl)
    with output_file.open("w") as out_f, txt_file.open() as in_f:
        pbar = tqdm.tqdm(enumerate(in_f))
        for line_num, line in pbar:

            words = line.strip().split(" ")
            all_subwords = [tokenizer.tokenize(word) for word in words]

            to_write = ""
            for subwords in all_subwords:
                found_subwords.update(subwords)
                if subword_sep in subwords:
                    raise Exception(
                        f"Subword separator '{subword_sep}' was found in the dataset at line {line_num}"
                    )
                to_write += subword_sep.join(subwords) + " "

            if not to_write:
                raise Exception(f"Found empty line in dataset at line {line_num}")
            else:
                # Remove extra " " at end
                to_write = to_write[:-1]

            out_f.write(to_write + "\n")

    print(f"Found {len(found_subwords)} subwords.")


def _oov_ratio(words_in_vocab: Set[str], text: List[str]) -> float:
    not_found = 0
    for word in text:
        if word not in words_in_vocab:
            not_found += 1
    return not_found / len(text)


def _acheive_oov_ratio(
    train_counter: Counter[str], dev_counter: Counter[str], desired_oov_ratio: float
) -> Set[str]:
    """Select a subset of the vocab words to achieve a desired OOV ratio in some text.

    That text is represented by text_counter.
    """
    new_vocab = set()
    in_vocab_count = 0
    cur_oov_ratio = 1.0
    dev_total = sum(dev_counter.values())
    dev_in_vocab_count = 0
    for word, train_count in train_counter.most_common():
        if train_count < 2:  # We want to have at least some OOV words
            break

        new_vocab.add(word)

        if word in dev_counter:
            dev_in_vocab_count += dev_counter[word]
            new_oov_ratio = (dev_total - dev_in_vocab_count) / dev_total
            if abs(desired_oov_ratio - new_oov_ratio) < abs(
                desired_oov_ratio - cur_oov_ratio
            ):
                cur_oov_ratio = new_oov_ratio
            else:
                break
    print(
        f"Was asked for OOV ratio: {desired_oov_ratio}. Could achieve OOV ratio of {cur_oov_ratio}."
    )
    return new_vocab


@app.command()
def make_vocab(
    output_pkl_file: Path,
    train_dev_pairs: List[Path] = typer.Argument(
        None,
        help="Pairs of training and dev files."
        " The train file will be where word counts are calculated."
        " The dev file is used to determine the number of words to include to approximate the "
        " desired ratio of uknown words in an unseen test set.",
    ),
    oov_ratio: float = typer.Option(
        0.15, help="The desired ratio of OOV count to total count in the dev set."
    ),
    subword_sep: Optional[str] = typer.Option(
        None,
        help="If this option is set, we'll asssume the input files are subword tokenized, where, "
        "in each word, this subword_sep value (for example, the '|' char) was used to separate "
        "subwords.",
    ),
) -> None:
    """Create a "Vocabulary" object (look at mlgvsnl/src/vocab.py) from given word
    tokenized files.

    An example run looks like:

        $ python data_prep.py vocab-from-word-files vocab.pkl train_caps.txt dev_caps.txt --unk-ratio=0.8

    One can also include multiple PAIRS of train and dev files:

        $ python data_prep.py vocab-from-word-files vocab.pkl zh/train_caps.txt zh/dev_caps.txt en/train_caps.txt en/dev_caps.txt --unk-ratio=0.8
    """

    if len(train_dev_pairs) % 2 != 0:
        raise Exception(
            f"Pairs of training and dev sets expected, but {len(train_dev_pairs)} is not an even "
            "number"
        )

    if subword_sep is None:
        split_pattern = r"\s|\n"  # Split by space or new line char
    else:
        split_pattern = re.escape(subword_sep) + "|" + r"\s+"

    final_words_from_each_file = {}

    train_files, dev_files = train_dev_pairs[0::2], train_dev_pairs[1::2]
    for train_file, dev_file in zip(train_files, dev_files):
        print(
            f"Inspecting traing file: {str(train_file)} and dev file {str(dev_file)}..."
        )
        with train_file.open() as f:
            words_in_train = re.split(split_pattern, f.read().lower())
            train_counter = Counter(words_in_train)
            print(
                f"Read total {len(words_in_train)} words({len(train_counter)} unique words) from {str(train_file)}."
            )

        # If, in the input files,  there are consecutive spaces, or spaces at end of
        # line, or other misc edge cases, take care of them.
        for bad in [" ", "", "\n"] + SPECIAL_TOKENS:
            if bad in train_counter:
                print(
                    f"Warning: '{bad}' found as a word. Maybe train file is malformed. Removing it and proceeding."
                )
                train_counter.pop(bad)

        with dev_file.open() as f:
            words_in_dev = re.split(split_pattern, f.read().lower())
            dev_counter = Counter(words_in_dev)

        final_vocab = _acheive_oov_ratio(train_counter, dev_counter, oov_ratio)
        final_words_from_each_file[train_file] = final_vocab

    lens = {
        str(train_file): len(vocab)
        for train_file, vocab in final_words_from_each_file.items()
    }
    print(
        f"Will output a final pickle with following num of subwords from each file: {lens}"
    )

    vocab = Vocabulary()

    for word in SPECIAL_TOKENS:
        vocab.add_word(word)

    for _, words_from_file in sorted(final_words_from_each_file.items()):
        for word in sorted(words_from_file):
            vocab.add_word(word)

    with output_pkl_file.open("wb") as fb:
        pkl.dump(vocab, fb)


class TransformersSingleTokenEmbedder:
    """Because Transofrmers library is not used to accepting pre-Bert-tokenized input."""

    def __init__(self, transformers_mdl: str, layer_num: int) -> None:
        self._transformers_mdl = transformers_mdl
        self._tokenizer = ts.BertTokenizer.from_pretrained(transformers_mdl)
        config = ts.BertConfig.from_pretrained(
            transformers_mdl, output_hidden_states=True, max_seq_length=3
        )

        self._model = ts.BertModel.from_pretrained(transformers_mdl, config=config)
        self._cls_token_id = self._tokenizer.convert_tokens_to_ids(
            self._tokenizer.cls_token
        )
        self._sep_token_id = self._tokenizer.convert_tokens_to_ids(
            self._tokenizer.sep_token
        )
        self._bert_unk_tok_id = self._tokenizer.convert_tokens_to_ids(
            self._tokenizer.unk_token
        )

        self._use_cuda = False
        self._layer_num = layer_num

        if torch.cuda.is_available():
            self._use_cuda = True
            self._model.cuda()

        print("Using model: ", self._model.config)

    @property
    def device(self) -> torch.device:
        """Return whether it's on GPU or CPU"""
        return next(self._model.parameters()).device  # type: ignore

    @property
    def dim(self) -> int:
        """Return the dimensions of the embeddings."""
        return self._model.config.hidden_size  # type: ignore

    def __call__(self, batch_bert_toks: List[str], debug: bool = False) -> torch.Tensor:
        """

        Args:
            batch_bert_toks: of length N.

        Output:
            tensor: A pytorch  tensor of size (N, BertDim)
                where BertDim refers to Bert's hidden dimensions'

        subword_batch]
        """
        input_ids = []

        for subword in batch_bert_toks:
            sent_input_ids = [
                self._cls_token_id,
                self._tokenizer.convert_tokens_to_ids(subword),
                self._sep_token_id,
            ]
            input_ids.append(sent_input_ids)


        pt_input_ids = torch.tensor(
            input_ids, device="cuda" if self._use_cuda else None
        )
        if debug:
            print("Subword batch: ", batch_bert_toks[:5])
            print("Input ids: ", input_ids[:5])
            print("Pytorch model input size:", pt_input_ids.size())

        with torch.no_grad():
            outputs = self._model(pt_input_ids)
        assert len(outputs) > 2
        _, _, hidden_states = outputs

        desired_layer_states = hidden_states[self._layer_num]

        assert desired_layer_states.size(1) == 3
        assert desired_layer_states.size(0) == len(batch_bert_toks)

        return desired_layer_states[:, 1]  # type: ignore # Ignore [CLS] and [SEP]


def _validate_vocab(vocab: Vocabulary) -> None:
    assert len(vocab.word2idx) == len(vocab.idx2word)
    for word, idx in vocab.word2idx.items():
        assert vocab.idx2word[idx] == word

    for tok in SPECIAL_TOKENS:
        if tok not in vocab.word2idx:
            raise Exception(f"Token ,'{tok}' not in vocab file!")
    if vocab.word2idx[SPECIAL_TOKENS[0]] != 0:
        raise Exception("The padding token MUST be at index 0. It's currently not.")

    first_words = [vocab.idx2word[i] for i in range(len(SPECIAL_TOKENS))]
    if sorted(first_words) != sorted(SPECIAL_TOKENS):
        raise Exception(
            f"The special tokens ({SPECIAL_TOKENS}) are not found at the beginning of "
            "the vocab(ie, their indices don't number 0 through {len(SPECIAL_TOKENS) -1}."
        )


def _read_vocab(vocab_pkl_file: Path) -> Vocabulary:
    with vocab_pkl_file.open("rb") as fb:
        vocab = pkl.load(fb)
    _validate_vocab(vocab)

    if not isinstance(vocab, Vocabulary):
        raise Exception("The vocab_pkl_file was not a Vocabulary object.")

    all_subwords = [vocab.idx2word[i] for i in range(len(vocab))]
    subword_preview = ",".join(['"' + s + '"' for s in all_subwords[:5]])
    if len(vocab) > 5:
        subword_preview += "..."
    print(f"Read vocab with {len(vocab)} subwords: {subword_preview}")
    return vocab


def _nltk_tree_to_paren(tree: Union[str, nltk.Tree]) -> str:
    """Convert benepar parse output to something like

    ( Man

    """

    if isinstance(tree, str):
        return tree

    subtree_results = [_nltk_tree_to_paren(subtree) for subtree in tree]

    if len(subtree_results) == 1:
        return subtree_results[0]

    return "( " + " ".join(subtree_results) + " )"


@app.command()
def parse_sents(
    test_caps_file: Path,
    output_ground_truth_file: Path,
    benepar_model_name: str,
    batch_size: int = 64,
) -> None:
    """Prepare gold ground truth using https://github.com/udnaan/self-attentive-parser.

    The above link is a fork of the original project that has not yet been merged. It has a fix that
    allows benepar to work with Tensorflow2, which is the default installation. You can install it
    with

        pip install git+https://github.com/udnaan/self-attentive-parser

    Args:
            test_caps_file: A text file where each word is separated by space, and one sentence per
                line.
            output_ground_truth_file:
            benepar_model_name: The name of the model to use. Find the right name from here:
                https://github.com/nikitakit/self-attentive-parser#available-models

                NOTE: You might have to do

                >>> import benpar
                >>> benepar.download('benepar_en2_large') # Or whatever other model you use
    """
    parser = benepar.Parser(benepar_model_name, batch_size=batch_size)

    with test_caps_file.open() as f:
        sents = [line.strip().split() for line in f]

    sents = [[word for word in sent if word] for sent in sents]  # Remove empty strings
    print(f"Parsing {len(sents)} sentences...")
    trees = list(parser.parse_sents(sents))
    print(f"Converting to the bracket format ...")
    paren_expr = [_nltk_tree_to_paren(tree) for tree in trees]

    with output_ground_truth_file.open("w") as f:
        f.writelines(line + "\n" for line in paren_expr)


@app.command()
def extract_word_embs(
    vocab_pkl_file: Path, output_npy_file: Path, all_embs_file: Path,
) -> None:
    """

    Extract the embedding for each word in the vocab from the given all_embs_file file and write it
    in a torch tensor format to output_npy_file.

    Args:
        vocab_pkl_file:
        output_npy_file:
        all_embs_file: The zip or gz  file containing word embeddings. Note that this
        has to contain a single file, where each line is the word, followed by the
        vector.  Like this:

            <word> 0.12 11.2 423.3 .....

        unk_tok:
    """
    vocab = _read_vocab(vocab_pkl_file)
    num_words = len(vocab)  # Read this before we modify the vocab below

    idx2vec = {}

    # Support both zip and gz. A zip file can possibly contain multiple files, but
    # we accept zip files containing only one file.
    # The gz file must be a compressed text file.
    if all_embs_file.suffix == ".zip":
        compressed_file = zipfile.ZipFile(all_embs_file)
        files_in_zip = compressed_file.infolist()
        if len(files_in_zip) != 1:
            raise Exception(
                f"Passed zip file contains {len(files_in_zip)} files. Expecting one."
            )

        binary_file = compressed_file.open(files_in_zip[0])
        text_file: TextIO = TextIOWrapper(binary_file)  # type: ignore[arg-type]
    elif all_embs_file.suffix == ".gz":
        if all_embs_file.stem.endswith(".tar"):
            raise Exception("We don't support .tar.gz currently, just .gz")
        text_file = gzip.open(all_embs_file, "rt")  # type: ignore[assignment]

    words_found_pbar: "tqdm.tqdm[None]" = tqdm.tqdm(
        total=len(vocab), desc="Words found.", position=0
    )

    # Make into set for faster lookup
    special_toks_set = set(SPECIAL_TOKENS)
    for line in tqdm.tqdm(text_file, desc="Words checked in embeddings file."):
        split = line.split()
        word = split[0]
        idx = vocab.word2idx.get(word)
        if word in special_toks_set or idx is None:
            continue
        vec = [float(i) for i in split[1:]]
        idx2vec[idx] = torch.tensor(vec)
        del vocab.idx2word[idx]
        del vocab.word2idx[word]
        words_found_pbar.update()

        if len(vocab.word2idx) == 1:  # Only unk tok left
            break

    text_file.close()
    if all_embs_file.suffix == ".zip":
        compressed_file.close()

    if len(vocab.idx2word) > 1:
        print(
            f"WARNING: Did not find vectors for {len(vocab.idx2word) - 1} words. Will set their vectors"
            " to the vector for the unk tok (which will be the average of all the vectors."
        )

    # Includes the unk_tok
    remaining_ids = list(vocab.idx2word)

    dim = len(next(iter(idx2vec.values())))  # Dimension of vectors
    embs = torch.zeros((num_words, dim), dtype=torch.float32)
    average_emb = torch.zeros((1, dim), dtype=torch.float32)

    for idx, vec in idx2vec.items():
        embs[idx] = vec
        average_emb += vec

    average_emb /= len(idx2vec)

    # Initalize vectors for  words that were not found in the vocabulary with the
    # same vector as UNK.
    for idx in vocab.idx2word:
        embs[idx] = average_emb

    with output_npy_file.open("wb") as fb:
        np.save(fb, embs.numpy())


@app.command()
def extract_subword_embs(
    vocab_pkl_file: Path,
    output_npy_file: Path,
    layer_num: int = 7,
    batch_size: int = 1000,
    transformers_mdl: str = "bert-base-multilingual-uncased",
) -> None:
    """

    Given a pickle file containing a Vocabulary object, extract from mBERT
    the embeddings for the word and write it in a .npy file (ie, call numpy.save() on
    the embedding matrix).

    The subwords are fed to bert in this way: [CLS] subword [SEP]

    Args:
        layer_num: The BERT layer to extract. Default is 7 because, according to this
            [meta-analysis paper](https://arxiv.org/abs/2002.12327), it is one of the
            most likely layers to encode syntax.

            NOTE: This is zero indexed, ie, the first layer is layer num 0.
        batch_size: The batch size used to group things.
    """

    vocab = _read_vocab(vocab_pkl_file)

    # Exclude unk_tok and pad_tok
    non_special_toks = iter(
        vocab.idx2word[idx] for idx in range(len(SPECIAL_TOKENS), len(vocab))
    )

    embeddings = []

    embedder = TransformersSingleTokenEmbedder(transformers_mdl, layer_num)

    for i, subword_batch in tqdm.tqdm(enumerate(chunked(non_special_toks, batch_size))):
        embs = embedder(subword_batch, debug=i < 5)
        embeddings.append(embs)

    non_special_tok_embs = torch.cat(embeddings)
    final_embs = torch.zeros((len(vocab), embedder.dim), device=embedder.device)

    # Set the embedding of the special toks to the average of all embeddings
    final_embs[: len(SPECIAL_TOKENS)] = non_special_tok_embs.mean(dim=0, keepdim=True)
    final_embs[len(SPECIAL_TOKENS) :] = non_special_tok_embs

    with output_npy_file.open("wb") as fb:
        np.save(fb, final_embs.cpu().numpy())


if __name__ == "__main__":
    app()
