from __future__ import annotations
from functools import cached_property
from random import Random
import pickle as pkl
import math
from typing import (
    Dict,
    overload,
    NamedTuple,
    Sequence,
    Sized,
    Iterator,
    Iterable,
    Any,
    Iterator,
    cast,
    Type,
    TYPE_CHECKING,
    Tuple,
    List,
    List,
    TypeVar,
    Generic,
)
from typing_extensions import Protocol
from pathlib import Path
import nltk  # type: ignore
from itertools import chain, accumulate
import abc
import numpy as np
import os

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
from coll_utils import Ordering

if TYPE_CHECKING:
    from vocab import Vocabulary


_Tco = TypeVar("_Tco", covariant=True)
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_T3 = TypeVar("_T3")
_T4 = TypeVar("_T4")
_T5 = TypeVar("_T5")


@overload
def myzip(__i: Sequence[Tuple[_T1]]) -> Tuple[List[_T1]]:
    ...


@overload
def myzip(__i: Sequence[Tuple[_T1, _T2]]) -> Tuple[List[_T1], List[_T2]]:
    ...


@overload
def myzip(
    __i: Sequence[Tuple[_T1, _T2, _T3]]
) -> Tuple[List[_T1], List[_T2], List[_T3]]:
    ...


@overload
def myzip(
    __i: Sequence[Tuple[_T1, _T2, _T3, _T4]]
) -> Tuple[List[_T1], List[_T2], List[_T3], List[_T4]]:
    ...


@overload
def myzip(
    __i: Sequence[Tuple[_T1, _T2, _T3, _T4, _T5]]
) -> Tuple[List[_T1], List[_T2], List[_T3], List[_T4], List[_T5]]:
    ...


def myzip(__i: Sequence[Tuple[Any, ...]]) -> Tuple[List[Any], ...]:
    """Wrap around zip(), and provide two additional benefits.

    1. Type inference for tuples of upto length 5.
    2. Returned items are lists, not tuples.
    """
    return tuple(list(col) for col in zip(*__i))


class SizedIndexedIterable(Sized, Iterable[_Tco], Protocol):
    def __getitem__(self, __idx: int) -> _Tco:
        ...


class ConcatSequence(Generic[_Tco]):

    seqs: SizedIndexedIterable[SizedIndexedIterable[_Tco]]

    def __init__(self, seqs: SizedIndexedIterable[SizedIndexedIterable[_Tco]]) -> None:
        self.seqs = seqs
        self._len = sum(map(len, seqs))

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[_Tco]:
        for i in range(len(self)):
            yield self[i]

    def get_detailed_idx(self, idx: int) -> Tuple[int, int]:
        if idx < 0:
            idx += len(self)

        for seq_num, seq in enumerate(self.seqs):
            if idx >= len(seq):
                idx -= len(seq)
            else:
                return (seq_num, idx)
        raise IndexError(
            f"{idx} is beyond the total len of this SeqOfSeqs (which is {len(self)}."
        )

    def __getitem__(self, idx: int) -> _Tco:
        seq_num, idx = self.get_detailed_idx(idx)
        return self.seqs[seq_num][idx]


class LanguageData(NamedTuple):
    vocab: Vocabulary
    dir_: Path


class VGSNLDataset(Dataset, Generic[_Tco]):
    @cached_property
    @abc.abstractmethod
    def subword(self) -> bool:
        pass

    @cached_property
    @abc.abstractmethod
    def lang_datas(self) -> Dict[Lang, LanguageData]:
        pass

    @abc.abstractmethod
    def collate_fn(self, exs: List[_Tco]) -> object:
        pass

    @abc.abstractmethod
    def __getitem__(self, __i: int) -> _Tco:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    def __iter__(self) -> Iterator[_Tco]:
        for i in range(len(self)):
            yield self[i]


Lang = str
WordExample = Tuple[Lang, Tuple[Tensor, List[int], int, int]]
SubwordExample = Tuple[Lang, Tuple[Tensor, List[List[int]], int, int]]
Example = TypeVar("Example", SubwordExample, WordExample)
Batch = Tuple[Lang, Tuple[Tensor, Tensor, List[int], List[int]]]


class SingleDataset(VGSNLDataset[Example]):
    """Handles image related dataset loading. Captions are handled by subclasses. Not
    meant to be instantiated directly.
    """

    _captions: List[List[Any]]
    _vocab: Vocabulary

    @cached_property
    def lang_datas(self) -> Dict[Lang, LanguageData]:
        return {self._lang: LanguageData(self._vocab, Path(self._data_path))}

    def __init__(
        self,
        data_path: str,
        data_split: str,
        vocab_filename: str,
        load_img: bool,
        img_dim: int,
    ):
        self._data_split = data_split
        self._data_path = data_path
        self._vocab_filename = vocab_filename

        with open(Path(self._data_path) / self._vocab_filename, "rb") as fb:
            self._vocab: Vocabulary = pkl.load(fb)

        self.prepare_text()

        # We assume the directory name is the language identifier.
        # This is actually used only when trianing on multiple languages with
        # ConcatDataset
        self._lang = Path(data_path).name

        generic_caps_per_img_fpath = Path(data_path) / "caps_per_img.txt"
        split_caps_per_img_fpath = Path(data_path) / f"{data_split}_caps_per_img.txt"

        if generic_caps_per_img_fpath.exists() == split_caps_per_img_fpath.exists():
            raise Exception(
                f"Exactly one of {str(split_caps_per_img_fpath)} or"
                f" {str(generic_caps_per_img_fpath)} must exist."
            )
        elif generic_caps_per_img_fpath.exists():
            split_caps_per_img_fpath = generic_caps_per_img_fpath

        with split_caps_per_img_fpath.open() as f:
            str_caps_per_img = [l.strip() for l in f]

        # Allow last lines to be empty
        while str_caps_per_img[-1] == "":
            str_caps_per_img.pop()

        self.load_img = load_img
        # image features
        # (DsetL, ImgD)
        if load_img:
            self.images = cast(
                "np.ndarray[np.float64]",
                np.load(os.path.join(data_path, f"{data_split}_ims.npy")),
            )

            # Allow one caption per image for entire dataset,
            # or a caption per image equal separately for each image.
            if len(str_caps_per_img) == 1:
                caps_per_img = [int(str_caps_per_img[0])] * len(self.images)
            else:
                if not len(str_caps_per_img) == len(self.images):
                    raise Exception(
                        f"caps_per_img file length (which is {len(str_caps_per_img)} is"
                        f"neither one, nor equal to the "
                        "number of images(which is {len(self.images)}."
                    )
                caps_per_img = list(map(int, str_caps_per_img))

            num_exp_caps = sum(caps_per_img)
            if not num_exp_caps == len(self._captions):
                raise Exception(
                    f"THe total number of captions ({len(self._captions)}) is not equal "
                    "to the value expected from caps_per_img.txt and the image features"
                    f" (that number is {num_exp_caps})"
                )

        else:
            caps_per_img = [1] * len(self._captions)
            # (DsetL, ImgD)
            self.images = np.zeros((len(self._captions), img_dim))

        self.caps_per_img = caps_per_img

        # Build the mapping of caption number to image number
        self.cap_idx_to_img_idx = {}
        cur_img_num = 0
        total_caps = 0
        for img_num, img_per_cap in enumerate(self.caps_per_img):
            total_caps += img_per_cap
            while cur_img_num < total_caps:
                self.cap_idx_to_img_idx[cur_img_num] = img_num
                cur_img_num += 1

    @abc.abstractmethod
    def prepare_text(self) -> None:
        pass

    @abc.abstractmethod
    def collate_fn(self, exs: List[Example]) -> Batch:
        pass

    def __len__(self) -> int:
        return len(self._captions)


class SubwordDataset(SingleDataset[SubwordExample]):
    """ load precomputed captions and image features, but using subword tokenization """

    _SUBWORD_SEP = "|"

    @cached_property
    def subword(self) -> bool:
        return True

    def prepare_text(self) -> None:
        self._captions = []

        caps_filename = f"{self._data_split}_caps_subword.txt"
        with open(os.path.join(self._data_path, caps_filename), "r") as f:
            for line in f:
                words = line.strip().lower().split()
                subwords = [word.split(self._SUBWORD_SEP) for word in words]
                self._captions.append(subwords)

        print(
            f"Read {len(self._captions)} captions from {caps_filename}. For example: "
        )
        for cap in self._captions[:5]:
            print(f"\t{cap}")

    def __getitem__(self, index: int) -> SubwordExample:
        # image
        img_id = self.cap_idx_to_img_idx[index]

        # (ImgD,)
        image = torch.tensor(self.images[img_id])

        # caption
        caption = [
            [self._vocab(token) for token in tokens]
            for tokens in [["<start>"]] + self._captions[index] + [["<end>"]]
        ]
        return self._lang, (image, caption, index, img_id)

    @staticmethod
    def collate_fn(exs: List[SubwordExample]) -> Batch:
        """ build mini-batch tensors from a list of (image, caption) tuples """

        images: Sequence[Tensor]
        langs, data = myzip(exs)
        if not len(set(langs)) == 1:
            raise Exception(f"This dataset is supposed to have only one language.")

        # sort a data list by caption length
        # I don't know why though, apart from getting the max seq length. But that is
        # doable without sorting easily. Going by "If it ain't broke, don't fix it".
        data = sorted(data, key=lambda x: len(x[1]), reverse=True)

        images, captions, ids, _ = myzip(data)

        # (B, ImgD)
        stacked_images = torch.stack(images, 0)

        max_seq_len = len(captions[0])
        max_num_subword_per_word = max(
            len(subwords_per_word) for subwords_per_word in chain(*captions)
        )

        # (B, ImgD, S)
        targets = torch.zeros(
            len(captions), max_seq_len, max_num_subword_per_word
        ).long()

        lengths = [len(cap) for cap in captions]
        for sent_num, sent in enumerate(captions):
            for word_num, word in enumerate(sent):
                end = len(word)
                targets[sent_num, word_num, :end] = torch.tensor(
                    word, device=stacked_images.device
                )
        lang = langs[0]
        return lang, (stacked_images, targets, lengths, ids)


class WordDataset(SingleDataset[WordExample]):
    """ load precomputed captions and image features """

    @cached_property
    def subword(self) -> bool:
        return True

    def prepare_text(self) -> None:

        self._captions = []
        with open(
            os.path.join(self._data_path, f"{self._data_split}_caps.txt"), "r"
        ) as f:
            for line in f:
                words = line.strip().lower().split()
                self._captions.append(words)
        print(
            f"Read {len(self._captions)} captions from {self._data_split}_caps.txt. For "
            "example: "
        )
        for cap in self._captions[:5]:
            print(f"\t{cap}")

    def __getitem__(self, index: int) -> WordExample:
        # image
        img_id = self.cap_idx_to_img_idx[index]

        # (ImgD,)
        image = torch.tensor(self.images[img_id])

        caption = [
            self._vocab(token)
            for token in ["<start>"] + self._captions[index] + ["<end>"]
        ]

        return self._lang, (image, caption, index, img_id)

    @staticmethod
    def collate_fn(exs: List[WordExample]) -> Batch:
        """ build mini-batch tensors from a list of (image, caption) tuples """

        langs, data = myzip(exs)
        if not len(set(langs)) == 1:
            raise Exception(f"This dataset is supposed to have only one language.")
        lang = langs[0]

        # sort a data list by caption length
        data.sort(key=lambda x: len(x[1]), reverse=True)

        images, captions, ids, _ = myzip(data)

        # (B, ImgD)
        stacked_images = torch.stack(images, 0)

        # (B, L)
        targets = torch.zeros(len(captions), len(captions[0])).long()
        lengths = [len(cap) for cap in captions]
        for i, cap in enumerate(captions):
            end = len(cap)
            targets[i, :end] = torch.tensor(cap[:end])

        return lang, (stacked_images, targets, lengths, ids)


class ConcatDataset(VGSNLDataset[Example]):

    seq_of_dsets: ConcatSequence[Example]

    @cached_property
    def subword(self) -> bool:
        return self._subword

    @cached_property
    def lang_datas(self) -> Dict[Lang, LanguageData]:
        result = {}
        for dset in self.seq_of_dsets.seqs:
            dset = cast(SingleDataset, dset)
            result.update(dset.lang_datas)
        return result

    @cached_property
    def langs(self) -> Ordering[Lang]:
        return self._langs

    def __init__(
        self,
        data_path: str,
        data_split: str,
        vocab_filename: str,
        subword: bool,
        load_img: bool,
        img_dim: int,
    ):
        dir_ = Path(data_path)

        if subword:
            dset_class: Type[SingleDataset] = SubwordDataset
        else:
            dset_class = WordDataset

        langs = []
        dsets: List[SingleDataset[Example]] = []
        for subdir in dir_.iterdir():
            if "vocab" in subdir.name:
                continue
            langs.append(subdir.name)
            dsets.append(
                dset_class(
                    str(subdir),
                    data_split,
                    vocab_filename,
                    load_img=load_img,
                    img_dim=img_dim,
                )
            )

        self.seq_of_dsets = ConcatSequence(tuple(dsets))
        self._langs = Ordering(langs)
        self._subword = subword

    def __getitem__(self, idx: int) -> Example:
        return self.seq_of_dsets[idx]

    def __len__(self) -> int:
        return sum(len(d) for d in self.seq_of_dsets.seqs)

    def collate_fn(self, exs: List[Example]) -> Batch:
        langs = set(pair[0] for pair in exs)
        if len(langs) != 1:
            raise RuntimeError(
                f"The sampler picked examples from different lanugages in the same"
                " batch."
            )
        lang = langs.pop()
        lang_idx = self.langs.indices[lang]
        dset_collate_fn = cast(
            SingleDataset[Example], self.seq_of_dsets.seqs[lang_idx]
        ).collate_fn
        batch = dset_collate_fn(exs)

        return batch


class DontMixBatchSampler(Sampler):
    def __init__(
        self,
        data_source: ConcatDataset[Any],
        batch_size: int = 128,
        shuffle: bool = True,
    ) -> None:
        batches_in_each = [
            math.ceil(len(dset) / batch_size) for dset in data_source.seq_of_dsets.seqs
        ]
        self._len = sum(batches_in_each)

        dset_ids_and_batch_ids = [
            (dset_id, batch_id)
            for dset_id, num_batches in enumerate(batches_in_each)
            for batch_id in range(num_batches)
        ]

        rnd = Random()
        if shuffle:
            rnd.shuffle(dset_ids_and_batch_ids)

        # This is one longer than number of sets
        offset_for_dsets = list(
            accumulate([0] + [len(d) for d in data_source.seq_of_dsets.seqs])
        )

        self.indices = []

        for dset_id, batch_id in dset_ids_and_batch_ids:
            dset_offset = offset_for_dsets[dset_id]
            example_offset = batch_id * batch_size

            first_id = dset_offset + example_offset

            # The last batch might be smaller than batch size
            # Also, this is one + the actual last id, per classic Python counting conventions
            last_id = min(first_id + batch_size, offset_for_dsets[dset_id + 1])

            batch_indices = list(range(first_id, last_id))
            if shuffle:
                rnd.shuffle(batch_indices)
            self.indices.append(batch_indices)

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self.indices)

    def __len__(self) -> int:
        return self._len


def is_concat_dset_path(dset_path: Path) -> bool:
    # Has sub directory wiht training files
    is_too = len(list(dset_path.glob("*/*_caps*.txt"))) != 0
    # Has training files as first level descendants
    is_not = len(list(dset_path.glob("*_caps*.txt"))) != 0

    if is_too and is_not:
        raise Exception(
            f"--data_path has training files at both "
            " top level and one level deep. Please choose one."
        )
    elif not (is_too or is_not):
        raise Exception(
            f"--data_path has no file matching '*_caps*.txt' at "
            "top level, nor does it have such a file at one level deep. "
            "Are you sure you typed it right?"
        )
    return is_too


def get_precomp_loader(
    data_path: str,
    data_split: str,
    vocab_filename: str,
    subword: bool,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 2,
    load_img: bool = True,
    img_dim: int = 2048,
) -> DataLoader[Example, Batch]:

    dset: VGSNLDataset[Any]
    if is_concat_dset_path(Path(data_path)):
        print(f"--data_path points to a concatenated dataset. Proceeding accrodingly.")
        dset = ConcatDataset(
            data_path, data_split, vocab_filename, subword, load_img, img_dim
        )
        batch_sampler = DontMixBatchSampler(dset, batch_size, shuffle)
        data_loader: DataLoader[Any, Batch] = DataLoader(
            dset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=dset.collate_fn,
            pin_memory=True,
        )
    else:
        dset_class: Type[SingleDataset[Any]]
        if subword:
            dset_class = SubwordDataset
        else:
            dset_class = WordDataset
        dset = dset_class(data_path, data_split, vocab_filename, load_img, img_dim)

        data_loader = DataLoader(
            dataset=dset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            collate_fn=dset.collate_fn,
        )
    return data_loader


def get_train_loaders(
    data_path: str,
    vocab_filename: str,
    batch_size: int,
    workers: int,
    subword: bool,
) -> Tuple[DataLoader[Example, Batch], DataLoader[Example, Batch]]:
    train_loader = get_precomp_loader(
        data_path=data_path,
        data_split="train",
        vocab_filename=vocab_filename,
        subword=subword,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
    )
    val_loader = get_precomp_loader(
        data_path=data_path,
        data_split="dev",
        vocab_filename=vocab_filename,
        subword=subword,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )

    return train_loader, val_loader  # type: ignore[return-value]


def get_eval_loader(
    data_path: str,
    split_name: str,
    vocab_filename: str,
    subword: bool,
    batch_size: int,
    workers: int,
    load_img: bool = False,
    img_dim: int = 2048,
) -> DataLoader[Example, Batch]:
    eval_loader = get_precomp_loader(
        data_path=data_path,
        data_split=split_name,
        vocab_filename=vocab_filename,
        subword=subword,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        load_img=load_img,
        img_dim=img_dim,
    )
    return eval_loader  # type: ignore[return-value]


##################### TESTS #################################


def test_seq_of_seqs() -> None:
    a = list(range(1, 5))
    b = list(range(5, 18))
    conc_seq = ConcatSequence([a, b])
    assert set(conc_seq) == set(a) | set(b)

    for idx, item in enumerate(chain(a, b)):
        assert conc_seq[idx] == item


def test_dont_mix_sampler() -> None:
    class Src(VGSNLDataset[_Tco]):
        def __init__(self, dsets: List[List[_Tco]]):
            self.seq_of_dsets = ConcatSequence(dsets)

        def __getitem__(self, idx: int) -> _Tco:
            return self.seq_of_dsets[idx]

        def __len__(self) -> int:
            return len(self.seq_of_dsets)

        collate_fn = None  # type: ignore
        lang_datas = None  # type: ignore
        subword = None  # type: ignore

    src = Src([list(range(5)), list(range(5, 18))])

    all_ = set(src)
    for bsize in [1, 3, 4, 6]:
        for shuffle in (True, False):
            sampler = DontMixBatchSampler(
                cast(ConcatDataset[Any], src), batch_size=bsize, shuffle=shuffle
            )

            total = set()
            print(f"shuffle={shuffle}, batchsize=", bsize, ":", end="")
            for batch in sampler:
                assert len(batch) > 0
                assert set(batch) - all_ == set()

                if batch[0] >= 5:
                    assert all(i >= 5 for i in batch)
                else:
                    assert all(i < 5 for i in batch)

                total.update(batch)
                print(batch, end=", ")

            assert len(total) == len(src)
            print("")


def test_concat_dset() -> None:
    import pytest
    import tqdm
    from typing import Counter

    data_dir = Path(__file__).parent.parent / "concat_dset_data"
    if not data_dir.exists():
        pytest.skip(f"Skipping test as {str(data_dir)} doesn't exist.")

    dset = ConcatDataset(
        str(data_dir),
        data_split="train",
        vocab_filename="vocab_subword.pkl",
        subword=True,
        load_img=True,
    )
    batch_sampler = DontMixBatchSampler(dset, batch_size=128)
    dloader: DataLoader[SubwordExample, Batch]
    dloader = DataLoader(
        dset, batch_sampler=batch_sampler, collate_fn=dset.collate_fn, num_workers=20
    )
    cnter: Counter[Lang] = Counter()
    pbar = tqdm.tqdm(dloader)
    total = 0
    for lang, _ in pbar:
        cnter.update([lang])
        total += 1
        # proportions = { l: c / total for l,c in cnter.items() }
    print(f"The dataloader had this num of batches per lang: ", cnter)
    assert set(dset.lang_datas.keys()) == set(cnter.keys())
