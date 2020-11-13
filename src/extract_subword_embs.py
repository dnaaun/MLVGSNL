from typing import List, Any, Counter
from pathlib import Path
from vocab import Vocabulary
import typer
import tqdm
from transformers.tokenization_bert import BertTokenizer

app = typer.Typer()

import abc
from typing import Generic, TypeVar

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
    txt_file: Path, output_file: Path = Path("./subword.txt"), subword_sep: str = "|",
) -> None:
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")

    found_subwords = set()
    num_subword_per_word = RunningAverage("num_subwords_per_word")

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
def make_vocabulary(
    subword_file: Path,
    output_file: Path = Path("subword-vocab.pkl"),
    subword_sep: str = "|",
    unk_thres: int = 0,
) -> None:
    subword_counter: Counter[str] = Counter()
    with subword_file.open() as in_f, output_file.open("wb") as out_fb:
        for line in tqdm.tqdm(in_f):
            for word in line.split(" "):
                subword_counter.update(word.split(subword_sep))


if __name__ == "__main__":
    app()
