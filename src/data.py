from typing import (
    Any,
    TYPE_CHECKING,
    Tuple,
    List,
    NewType,
    List,
    TypeVar,
    Generic,
    overload,
    Union,
)
from typing_extensions import Literal
import nltk  # type: ignore
from itertools import chain
import abc
import numpy as np
import os

import torch
from torch import Tensor
import torch.utils.data as data
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from vocab import Vocabulary


_T = TypeVar("_T")

_WordExample = Tuple[Tensor, Tensor, int, int]
_SubwordExample = Tuple[Tensor, List[List[int]], int, int]
_Example = TypeVar("_Example", _WordExample, _SubwordExample)

PrecompDsetBatch = Tuple[Tensor, Tensor, List[int], List[int]]


class PrecompDsetBase(data.Dataset, Generic[_Example]):
    """Handles image related dataset loading. Captions are handled by subclasses. Not meant
    to be instantiated directly.
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

        with open(os.path.join(data_path, "caps_per_img.txt")) as f:
            caps_per_img = int(f.read().strip())

        # image features
        # (DsetL, ImgD)
        if load_img:
            self.images: "np.ndarray[np.float64]" = np.load(
                os.path.join(data_path, f"{data_split}_ims.npy")
            )
        else:
            # (DsetL, ImgD)
            self.images = np.zeros((len(self) // caps_per_img, img_dim))

        assert self.images.shape[0] * caps_per_img == len(self)
        self.caps_per_img = caps_per_img

    @abc.abstractmethod
    def prepare_captions(self) -> None:
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
        img_id = index // self.caps_per_img

        # (ImgD,)
        image = torch.tensor(self.images[img_id])

        # caption
        caption = [
            [self.vocab(token) for token in tokens]
            for tokens in [["<start>"]] + self.captions[index] + [["<end>"]]
        ]
        return image, caption, index, img_id

    @staticmethod
    def collate_fn(data: List[_SubwordExample]) -> PrecompDsetBatch:
        """ build mini-batch tensors from a list of (image, caption) tuples """
        # sort a data list by caption length
        # I don't know why though, apart from getting the max seq length. But that is doable
        # without sorting easily. Going by "If it ain't broke, don't fix it".
        data.sort(key=lambda x: len(x[1]), reverse=True)

        images: List[Tensor]
        captions: List[List[List[int]]]
        ids: List[int]
        img_ids: List[int]
        images, captions, ids, img_ids = (list(_) for _ in zip(*data))

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
            f"Read {len(self.captions)} captions from {self.data_split}_caps.txt. For example: "
        )
        for cap in self.captions[:5]:
            print(f"\t{cap}")

    def __getitem__(self, index: int) -> _WordExample:
        # image
        img_id = index // self.caps_per_img

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
    def collate_fn(data: List[_WordExample]) -> PrecompDsetBatch:
        """ build mini-batch tensors from a list of (image, caption) tuples """
        # sort a data list by caption length
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions, ids, img_ids = (list(_) for _ in zip(*data))

        # (B, ImgD)
        stacked_images = torch.stack(images, 0)

        # (B, L)
        targets = torch.zeros(len(captions), len(captions[0])).long()
        lengths = [len(cap) for cap in captions]
        for i, cap in enumerate(captions):
            end = len(cap)
            targets[i, :end] = cap[:end]

        return stacked_images, targets, lengths, ids


if TYPE_CHECKING:
    PrecompDLoader = data.DataLoader[_Example, PrecompDsetBatch]


@overload
def get_precomp_loader(
    data_path: str,
    data_split: str,
    vocab: "Vocabulary",
    subword: Literal[False],
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 2,
    load_img: bool = True,
    img_dim: int = 2048,
) -> "PrecompDLoader[_Example]":
    ...


@overload
def get_precomp_loader(
    data_path: str,
    data_split: str,
    vocab: "Vocabulary",
    subword: Literal[True],
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 2,
    load_img: bool = True,
    img_dim: int = 2048,
) -> "PrecompDLoader[_Example]":
    ...


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
) -> "PrecompDLoader[_Example]":
    dset: Union[PrecompWordDataset, PrecompSubwordDataset]

    if subword:
        dset = PrecompSubwordDataset(data_path, data_split, vocab, load_img, img_dim)
    else:
        dset = PrecompWordDataset(data_path, data_split, vocab, load_img, img_dim)
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
) -> Tuple["PrecompDLoader[_Example]", "PrecompDLoader[_Example]"]:
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
