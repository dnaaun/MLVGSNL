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
from pathlib import Path
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

        generic_caps_per_img_fpath = Path(data_path) /  "caps_per_img.txt"
        split_caps_per_img_fpath = Path(data_path) /  f"{data_split}_caps_per_img.txt"

        if generic_caps_per_img_fpath.exists() == split_caps_per_img_fpath.exists():
            raise Exception(f"Exactly one of {str(split_caps_per_img_fpath)} or {str(generic_caps_per_img_fpath)}"
                    " must exist."
                    )
        elif generic_caps_per_img_fpath.exists():
            split_caps_per_img_fpath = generic_caps_per_img_fpath

        with split_caps_per_img_fpath.open() as f:
            str_caps_per_img = [l.strip() for l in f]

        # Allow last lines to be empty
        while str_caps_per_img[-1] == "":
            str_caps_per_img.pop()

        # image features
        # (DsetL, ImgD)
        if load_img:
            self.images: "np.ndarray[np.float64]" = np.load(
                os.path.join(data_path, f"{data_split}_ims.npy")
            )

            # Allow one caption per image for entire dataset,
            # or a caption per image equal separately for each image.
            if len(str_caps_per_img) == 1:
                caps_per_img = [int(str_caps_per_img[0])] * len(self.images)
            else:
                if not len(str_caps_per_img) == len(self.images):
                    raise Exception(
                        f"caps_per_img file length (which is {len(str_caps_per_img)} is neither one, nor equal to the "
                        f"number of images(which is {len(self.images)}."
                    )
                caps_per_img = list(map(int, str_caps_per_img))

            if not (num_exp_caps := sum(caps_per_img)) == len(self.captions):
                raise Exception(
                    f"THe total number of captions ({len(self.captions)}) is not equal to the value"
                    " expected from caps_per_img.txt and the image features (that number is "
                    f" {num_exp_caps})"
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
