from typing import List, Any, Counter, Set
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
import abc
from typing import Generic, TypeVar, TYPE_CHECKING, Tuple, Union
import zipfile

import lazy_import

if TYPE_CHECKING:  # These two modules take a long time so do lazy importing.
    import nltk  # type: ignore
    import transformers as ts
    import torch
    import benepar  # type: ignore
else:
    nltk = lazy_import.lazy_module("nltk")
    ts = lazy_import.lazy_module("transformers")
    torch = lazy_import.lazy_module("torch")
    benepar = lazy_import.lazy_module("benepar")


# Add the path of the mlvgnsl code so we can import Vocabulary.
import sys

src_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(src_path)

from src.vocab import Vocabulary


app = typer.Typer()

_T = TypeVar("_T")


class RunningMetric(abc.ABC, Generic[_T]):

    _value: _T

    def __init__(self, name: str) -> None:
        self._name = name
        self._steps = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._name}): {self._value}"

    def update(self, val: _T) -> None:
        self._steps += 1
        self._update(val)

    @abc.abstractmethod
    def _update(self, val: _T) -> None:
        pass


class RunningAverage(RunningMetric[float]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._total_sum = 0.0
        self._value = 0

    def _update(self, val: float) -> None:
        self._total_sum += val
        self._value = self._total_sum / self._steps


@app.command()
def convert_to_subword(
    txt_file: Path,
    output_file: Path = Path("./subword-vocab.txt"),
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

    found_subwords = set()
    num_subword_per_word = RunningAverage("num_subwords_per_word")

    tokenizer = ts.BertTokenizer.from_pretrained(transformers_mdl)
    with output_file.open("w") as out_f, txt_file.open() as in_f:
        pbar = tqdm.tqdm(enumerate(in_f))
        for line_num, line in pbar:

            # Update us with the average subword number sometimes
            if line_num % 1000 == 0:
                pbar.set_description(repr(num_subword_per_word))
            words = line.strip().split(" ")
            all_subwords = [tokenizer.tokenize(word) for word in words]

            to_write = ""
            for subwords in all_subwords:
                found_subwords.update(subwords)
                num_subword_per_word.update(len(subwords))
                if subword_sep in subwords:
                    raise Exception(
                        f"Subword separator '{subword_sep}' was found in the dataset at line {line_num}"
                    )
                to_write += "|".join(subwords) + " "

            if not to_write:
                raise Exception(f"Found empty line in dataset at line {line_num}")
            else:
                # Remove extra " " at end
                to_write = to_write[:-1]

            out_f.write(to_write + "\n")

    print(f"Found {len(found_subwords)} subwords.")


@app.command()
def vocab_from_word_files(
    output_pkl_file: Path,
    input_files: List[Path],
    unk_cutoff: float = 0.85,
    char_level: bool = False,
) -> None:
    """Create a "Vocabulary" object (look at mlgvsnl/src/vocab.py) from given word
    tokenized files.

    Args:
        output_pkl_file: The pkl file to write the output to.
        input_files: A list of files, each of htem has to look like this:

            A restaurant has modern wooden tables and chairs .
            A long restaurant table with rattan rounded back chairs .
            a long table with a plant on top of it surrounded with wooden chairs
            A long table with a flower arrangement in the middle for meetings

        subword_sep: The subword separator used inthe input files (above, it's the '|' char).
        unk_cutoff: What percentage of the unique subwords to keep in the final
            vocabulary.  Note that, for _each_ input file the specified percentage of unique
            subwords is retained. This is to avoid one big input file dominating all the
            rest in the final output.
        do_char_level: This specifies if the text file is character based (as opposed to word based).
            For example, Flickr30K Chinese has no space characters, each character is a "word"
            by itself.

    REMEBMER: Never include the development set, or the test set, when making the vocabulary file.
    """

    subwords_per_inp_file = {}
    if char_level:
        split_pattern = "|\n"  # Split by "empty space" or a new line char
    else:
        split_pattern = r"\s|\n"  # Split by space or new line char

    if len(set(input_files)) != len(input_files):
        raise Exception(f"Duplicates found in given input files: {input_files}")

    for inp_file in tqdm.tqdm(input_files, desc="files processed."):
        counter = Counter[str]()
        with inp_file.open() as f:
            words = set(re.split(split_pattern, f.read().lower()))
            print(f"Read {len(words)} words from {str(inp_file)}.")
            counter.update(words)

        for bad in [" ", "", "\n"]:
            if bad in counter:
                print(
                    f"Warning: '{bad}' found. Maybe input file is malformed."
                    " Removing it and proceeding."
                )

                counter.pop(bad)

        selected_words: Set[str] = set()
        desired_num = unk_cutoff * len(counter)
        most_common_iter = iter(pair[0] for pair in counter.most_common())
        while len(selected_words) < desired_num:
            selected_words.add(next(most_common_iter))
        print(f"Some selected words: ", list(islice(selected_words, 0, 5)))
        subwords_per_inp_file[inp_file] = selected_words

    lens = {
        str(inp_file): len(subwords)
        for inp_file, subwords in subwords_per_inp_file.items()
    }
    print(
        f"Will output a final pickle with following num of subwords from each file: {lens}"
    )

    vocab = Vocabulary()
    vocab.add_word("<unk>")
    for subword in chain(*subwords_per_inp_file.values()):
        vocab.add_word(subword)

    with output_pkl_file.open("wb") as fb:
        pkl.dump(vocab, fb)


@app.command()
def vocab_from_subword_files(
    output_pkl_file: Path,
    input_files: List[Path],
    unk_cutoff: float = 0.85,
    subword_sep: str = "|",
) -> None:
    """Create a "Vocabulary" object (look at mlgvsnl/src/vocab.py) from given subword
    tokenized files.

    Args:
        output_pkl_file: The pkl file to write the output to.
        input_files: A list of files, each of htem has to look like this:

            a restaurant has modern wooden tables and chair|##s .
            a long restaurant table with ratt|##an rounded back chair|##s .
            a long table with a plant on top of it surrounded with wooden chair|##s
            a long table with a flower arrangement in the middle for meetings

        subword_sep: The subword separator used inthe input files (above, it's the '|'
            char).
        unk_cutoff: What percentage of the unique subwords to keep in the final
        vocabulary.  Note that, for _each_ input file the specified percentage of unique
        subwords is retained. This is to avoid one big input file dominating all the
        rest in the final output.
    """

    subwords_per_inp_file = {}
    split_pattern = re.escape(subword_sep) + "|" + r"\s+"

    if len(set(input_files)) != len(input_files):
        raise Exception(f"Duplicates found in given input files: {input_files}")

    for inp_file in tqdm.tqdm(input_files, desc="files processed."):
        counter = Counter[str]()
        with inp_file.open() as f:
            subwords = re.split(split_pattern, f.read().lower())
            print(f"Read {len(subwords)} unique subwords from {str(inp_file)}.")
            counter.update(subwords)

        for bad in [" ", "", "\n"]:
            if bad in counter:
                print(
                    f"Warning: '{bad}' found. Maybe input file is malformed."
                    " Removing it and proceeding."
                )

                counter.pop(bad)

        selected_subwords: Set[str] = set()
        desired_num = unk_cutoff * len(counter)
        most_common_subwords = iter(pair[0] for pair in counter.most_common())
        while len(selected_subwords) < desired_num:
            selected_subwords.add(next(most_common_subwords))
        subwords_per_inp_file[inp_file] = selected_subwords

    lens = {
        str(inp_file): len(subwords)
        for inp_file, subwords in subwords_per_inp_file.items()
    }
    print(
        f"Will output a final pickle with following num of subwords from each file: {lens}"
    )

    vocab = Vocabulary()
    vocab.add_word("<unk>")
    for subword in chain(*subwords_per_inp_file.values()):
        vocab.add_word(subword)

    with output_pkl_file.open("wb") as fb:
        pkl.dump(vocab, fb)


class BertSingleTokenEmbedder:
    """Because Transofrmers library is not used to accepting pre-Bert-tokenized input."""

    def __init__(self, bert_mdl: str, layer_num: int) -> None:
        self._tokenizer = ts.BertTokenizer.from_pretrained(bert_mdl)
        config = ts.BertConfig.from_pretrained(
            bert_mdl, output_hidden_states=True, max_seq_length=3
        )

        self._model = ts.BertModel.from_pretrained(bert_mdl, config=config)
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

    def __call__(self, batch_bert_toks: List[str], debug: bool = False) -> torch.Tensor:
        """

        Args:
            batch_bert_toks: of length N.

        Output:
            tensor: A pytorch  tensor of size (N, BertDim)
                where BertDim refers to Bert's hidden dimensions'

        subword_batch]
        """
        input_ids = [
            [
                self._cls_token_id,
                self._tokenizer.convert_tokens_to_ids(subword),
                self._sep_token_id,
            ]
            for subword in batch_bert_toks
        ]

        pt_input_ids = torch.tensor(
            input_ids, device="cuda" if self._use_cuda else None
        )
        if debug:
            print("Subword batch: ", batch_bert_toks[:5])
            print("Input ids: ", input_ids[:5])
            print("Pytorch model input size:", pt_input_ids.size())

        outputs = self._model(pt_input_ids)
        assert len(outputs) > 2
        _, _, hidden_states = outputs

        desired_layer_states = hidden_states[self._layer_num]

        assert desired_layer_states.size(1) == 3
        assert desired_layer_states.size(0) == len(batch_bert_toks)

        return desired_layer_states[:, 1]  # type: ignore # Ignore [CLS] and [SEP]


def _read_vocab(vocab_pkl_file: Path) -> Vocabulary:
    with vocab_pkl_file.open("rb") as fb:
        vocab = pkl.load(fb)

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
    vocab_pkl_file: Path,
    output_torch_file: Path,
    word_emb_zip: Path,
    unk_tok: str = "<unk>",
) -> None:
    """

    Extract the embedding for each word in the vocab from the given word_emb_zip file and write it
    in a torch tensor format to output_torch_file.

    Args:
        vocab_pkl_file:
        output_torch_file:
        word_emb_zip: The zip file containing word embeddings. Note that this has to contain a
        single file, where each line is the word, followed by the vector.  Like this:

            <word> 0.12 11.2 423.3 .....

        unk_tok:
    """
    vocab = _read_vocab(vocab_pkl_file)
    num_words = len(vocab)  # Read this before we modify the vocab below

    if unk_tok not in vocab.word2idx:
        raise Exception(
            f"Unk tok '{unk_tok} not in vocab file! Please pass a different unk tok"
            " command line parameter. Or a different vocab file."
        )

    idx2vec = {}

    with zipfile.ZipFile(word_emb_zip) as word_emb_zip_flie:
        files_in_zip = word_emb_zip_flie.infolist()
        if len(files_in_zip) != 1:
            raise Exception(
                f"Passed zip file contains {len(files_in_zip)} files. Expecting one."
            )

        with word_emb_zip_flie.open(files_in_zip[0]) as fb:  # Opens it as binary file
            f: "io.TextIO" = TextIOWrapper(fb)  # type: ignore

            pbar = tqdm.tqdm(total=len(vocab), desc="Words found.")
            for line in f:
                split = line.split()
                word = split[0]
                idx = vocab.word2idx.get(word)
                if word == unk_tok or idx is None:
                    continue
                vec = [float(i) for i in split[1:]]
                idx2vec[idx] = torch.tensor(vec)
                del vocab.idx2word[idx]
                del vocab.word2idx[word]
                pbar.update()

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

    for idx in vocab.idx2word:
        embs[idx] = average_emb

    with output_torch_file.open("wb") as fb:
        torch.save(embs, fb)


@app.command()
def extract_subword_embs(
    vocab_pkl_file: Path,
    output_torch_file: Path,
    layer_num: int = 7,
    unk_tok: str = "<unk>",
    batch_size: int = 1000,
    transformers_mdl: str = "bert-base-multilingual-uncased",
) -> None:
    """

    Given a pickle file containing a Vocabulary object, extract from mBERT
    the embeddings for the word and write it in a .pth file (ie, call torch.save() on
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

    try:
        unk_tok_i = all_subwords.index(unk_tok)
        if unk_tok_i != 0:
            raise Exception("The code doesn't support unk_tok not being index 0.")
    except ValueError:
        raise Exception("UNK token not found!")

    all_subwords_except_unk = islice(all_subwords, 1, len(all_subwords))

    embeddings = []

    embedder = BertSingleTokenEmbedder(transformers_mdl, layer_num)

    for i, subword_batch in tqdm.tqdm(
        enumerate(chunked(all_subwords_except_unk, batch_size))
    ):
        embs = embedder(subword_batch, debug=i < 5)
        embeddings.append(embs)
        gc.collect()
        if i > 150:
            break

    all_embs = torch.cat(embeddings)

    with output_torch_file.open("wb") as fb:
        torch.save(all_embs, fb)


if __name__ == "__main__":
    app()
