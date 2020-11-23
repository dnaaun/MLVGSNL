from random import Random
import pickle as pkl
import math
from typing import (
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
    Union,
)
from typing_extensions import Literal, Protocol
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


_WordExample = Tuple[Tensor, Tensor, int, int]
_SubwordExample = Tuple[Tensor, List[List[int]], int, int]
_Example = TypeVar("_Example", _WordExample, _SubwordExample, covariant=True)

PrecompDsetBatch = Tuple[Tensor, Tensor, List[int], List[int]]

_Tco = TypeVar("_Tco", covariant=True)


class SizedIndexedIterable(Sized, Iterable[_Tco], Protocol):
    def __getitem__(self, __idx: int) -> _Tco:
        ...


class MyDset(Dataset, Generic[_Tco]):
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


_S = TypeVar("_S", bound="ConcatSequence")


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

    def __getitem__(self: _S, idx: int) -> _Tco:
        seq_num, idx = self.get_detailed_idx(idx)
        return self.seqs[seq_num][idx]


class PrecompDsetBase(MyDset[_Example]):
    """Handles image related dataset loading. Captions are handled by subclasses. Not
    meant to be instantiated directly.
    """

    captions: List[List[Any]]
    subword: bool

    def __init__(
        self,
        data_path: str,
        data_split: str,
        vocab: "Vocabulary",
        load_img: bool = True,
        img_dim: int = 2048,
    ):
        self.data_split = data_split
        self.data_path = data_path
        self.vocab = vocab
        self.prepare_captions()

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
            if not num_exp_caps == len(self.captions):
                raise Exception(
                    f"THe total number of captions ({len(self.captions)}) is not equal "
                    "to the value expected from caps_per_img.txt and the image features"
                    f" (that number is {num_exp_caps})"
                )

        else:
            caps_per_img = [1] * len(self.captions)
            # (DsetL, ImgD)
            self.images = np.zeros((len(self.captions), img_dim))

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
    def prepare_captions(self) -> None:
        pass

    @abc.abstractmethod
    def collate_fn(self, exs: List[_Example]) -> PrecompDsetBatch:
        pass

    def __len__(self) -> int:
        return len(self.captions)


class PrecompSubwordDataset(PrecompDsetBase[_SubwordExample]):
    """ load precomputed captions and image features, but using subword tokenization """

    _SUBWORD_SEP = "|"
    subword = True

    def prepare_captions(self) -> None:
        self.captions = []

        caps_filename = f"{self.data_split}_caps_subword.txt"
        with open(os.path.join(self.data_path, caps_filename), "r") as f:
            for line in f:
                words = line.strip().lower().split()
                subwords = [word.split(self._SUBWORD_SEP) for word in words]
                self.captions.append(subwords)

        print(f"Read {len(self.captions)} captions from {caps_filename}. For example: ")
        for cap in self.captions[:5]:
            print(f"\t{cap}")

    def __getitem__(self, index: int) -> _SubwordExample:
        # image
        img_id = self.cap_idx_to_img_idx[index]

        # (ImgD,)
        image = torch.tensor(self.images[img_id])

        # caption
        caption = [
            [self.vocab(token) for token in tokens]
            for tokens in [["<start>"]] + self.captions[index] + [["<end>"]]
        ]
        return image, caption, index, img_id

    @staticmethod
    def collate_fn(exs: List[_SubwordExample]) -> PrecompDsetBatch:
        """ build mini-batch tensors from a list of (image, caption) tuples """
        # sort a data list by caption length
        # I don't know why though, apart from getting the max seq length. But that is
        # doable without sorting easily. Going by "If it ain't broke, don't fix it".
        exs.sort(key=lambda x: len(x[1]), reverse=True)

        images: List[Tensor]
        captions: List[List[List[int]]]
        ids: List[int]
        images, captions, ids, _ = (list(_) for _ in zip(*exs))

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
        return stacked_images, targets, lengths, ids


class PrecompWordDataset(PrecompDsetBase[_WordExample]):
    """ load precomputed captions and image features """

    subword = False

    def prepare_captions(self) -> None:
        # captions
        self.captions = []
        with open(
            os.path.join(self.data_path, f"{self.data_split}_caps.txt"), "r"
        ) as f:
            for line in f:
                words = line.strip().lower().split()
                self.captions.append(words)
        print(
            f"Read {len(self.captions)} captions from {self.data_split}_caps.txt. For "
            "example: "
        )
        for cap in self.captions[:5]:
            print(f"\t{cap}")

    def __getitem__(self, index: int) -> _WordExample:
        # image
        img_id = self.cap_idx_to_img_idx[index]

        # (ImgD,)
        image = torch.tensor(self.images[img_id])

        caption = [
            self.vocab(token)
            for token in ["<start>"] + self.captions[index] + ["<end>"]
        ]

        # (l,)
        torch_caption = torch.tensor(caption)
        return image, torch_caption, index, img_id

    @staticmethod
    def collate_fn(exs: List[_WordExample]) -> PrecompDsetBatch:
        """ build mini-batch tensors from a list of (image, caption) tuples """
        # sort a data list by caption length
        exs.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions, ids, _ = (list(_) for _ in zip(*exs))

        # (B, ImgD)
        stacked_images = torch.stack(images, 0)

        # (B, L)
        targets = torch.zeros(len(captions), len(captions[0])).long()
        lengths = [len(cap) for cap in captions]
        for i, cap in enumerate(captions):
            end = len(cap)
            targets[i, :end] = cap[:end]

        return stacked_images, targets, lengths, ids


class ConcatDset(MyDset[Tuple[str, _Example]]):

    seq_of_dsets: ConcatSequence[_Example]
    lang_ordering: Ordering[str]
    subword: bool

    def __init__(
        self,
        data_path: str,
        data_split: str,
        vocab: "Vocabulary",
        subword: bool,
        load_img: bool = True,
        img_dim: int = 2048,
    ):
        dir_ = Path(data_path)

        if subword:
            dset_class: Type[PrecompDsetBase] = PrecompSubwordDataset
        else:
            dset_class = PrecompWordDataset

        langs = []
        dsets: List[PrecompDsetBase[_Example]] = []
        for subdir in dir_.iterdir():
            if "vocab" in subdir.name:
                continue
            langs.append(subdir.name)
            dsets.append(dset_class(str(subdir), data_split, vocab))

        self.seq_of_dsets = ConcatSequence(tuple(dsets))
        self.lang_ordering = Ordering(langs)

    def __getitem__(self, idx: int) -> Tuple[str, _Example]:
        lang_idx, idx = self.seq_of_dsets.get_detailed_idx(idx)
        lang = self.lang_ordering[lang_idx]
        return lang, self.seq_of_dsets.seqs[lang_idx][idx]

    def __len__(self) -> int:
        return sum(len(d) for d in self.seq_of_dsets.seqs)

    def collate_fn(
        self, exs: List[Tuple[str, _Example]]
    ) -> Tuple[str, PrecompDsetBatch]:
        langs, data = cast(Tuple[Tuple[str, ...], Tuple[_Example, ...]], zip(*exs))
        if len(set(langs)) != 1:
            raise RuntimeError(
                f"The sampler picked examples from different lanugages in the same"
                " batch."
            )
        lang = langs[0]
        lang_idx = self.lang_ordering.indices[lang]
        dset_collate_fn = cast(
            PrecompDsetBase[_Example], self.seq_of_dsets.seqs[lang_idx]
        ).collate_fn
        batch = dset_collate_fn(list(data))

        return (lang, batch)


class DontMixBatchSampler(Sampler):
    def __init__(
        self, data_source: ConcatDset[Any], batch_size: int = 128, shuffle: bool = True
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


def test_seq_of_seqs() -> None:
    a = list(range(1, 5))
    b = list(range(5, 18))
    conc_seq = ConcatSequence([a, b])
    assert set(conc_seq) == set(a) | set(b)

    for idx, item in enumerate(chain(a, b)):
        assert conc_seq[idx] == item


def get_precomp_loader(
    data_path: str,
    data_split: str,
    vocab: "Vocabulary",
    subword: bool,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 2,
    load_img: bool = True,
    img_dim: int = 2048,
) -> DataLoader:
    dset_class: Union[Type[PrecompWordDataset], Type[PrecompSubwordDataset]]

    if subword:
        dset_class = PrecompSubwordDataset
    else:
        dset_class = PrecompWordDataset

    dset = dset_class(data_path, data_split, vocab, load_img, img_dim)
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
    vocab: "Vocabulary",
    batch_size: int,
    workers: int,
    subword: Union[Literal[True], Literal[False]],
) -> Tuple[DataLoader, DataLoader]:
    train_loader = get_precomp_loader(
        data_path=data_path,
        data_split="train",
        vocab=vocab,
        subword=subword,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
    )
    val_loader = get_precomp_loader(
        data_path=data_path,
        data_split="dev",
        vocab=vocab,
        subword=subword,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )

    return train_loader, val_loader


def get_eval_loader(
    data_path: str,
    split_name: str,
    vocab: "Vocabulary",
    subword: Union[Literal[True], Literal[False]],
    batch_size: int,
    workers: int,
    load_img: bool = False,
    img_dim: int = 2048,
) -> DataLoader:
    eval_loader = get_precomp_loader(
        data_path=data_path,
        data_split=split_name,
        vocab=vocab,
        subword=subword,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        load_img=load_img,
        img_dim=img_dim,
    )
    return eval_loader


##################### TESTS #################################


def test_dont_mix_sampler() -> None:
    class Src(MyDset[_Tco]):
        def __init__(self, dsets: List[List[_Tco]]):
            self.seq_of_dsets = ConcatSequence(dsets)

        def __getitem__(self, idx: int) -> _Tco:
            return self.seq_of_dsets[idx]

        def __len__(self) -> int:
            return len(self.seq_of_dsets)

        collate_fn = None  # type: ignore

    src = Src([list(range(5)), list(range(5, 18))])

    all_ = set(src)
    for bsize in [1, 3, 4, 6]:
        for shuffle in (True, False):
            sampler = DontMixBatchSampler(
                cast(ConcatDset[Any], src), batch_size=bsize, shuffle=shuffle
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

    data_dir = Path(__file__).parent.parent / "concat_dset_data"
    if not data_dir.exists():
        pytest.skip(f"Skipping test as {str(data_dir)} doesn't exist.")

    with open(data_dir / "vocab_subword.pkl", "rb") as fb:
        vocab = pkl.load(fb)
    dset = ConcatDset(
        str(data_dir), data_split="train", vocab=vocab, subword=True, load_img=True
    )
    batch_sampler = DontMixBatchSampler(dset, batch_size=128)
    dloader = DataLoader(
        dset,
        batch_sampler=batch_sampler,
        collate_fn = dset.collate_fn
    )
    item = next(iter(dloader))
    breakpoint()
