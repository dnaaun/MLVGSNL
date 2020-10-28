from typing import Any, TYPE_CHECKING, Tuple, List, NewType, List
import nltk  # type: ignore
import numpy as np  # type: ignore
import os

import torch
import torch.utils.data as data

if TYPE_CHECKING:
    from vocab import Vocabulary

if TYPE_CHECKING:
    Base = data.Dataset[Any]
else:
    Base = data.Dataset


class PrecompDataset(Base):
    """ load precomputed captions and image features """

    def __init__(
        self,
        data_path: str,
        data_split: str,
        vocab: "Vocabulary",
        load_img: bool = True,
        img_dim: int = 2048,
    ):
        self.vocab = vocab

        with open(os.path.join(data_path, 'caps_per_img.txt')) as f:
            caps_per_img = int(f.read().strip())

        # captions
        self.captions = []
        with open(os.path.join(data_path, f"{data_split}_caps.txt"), "r") as f:
            for line in f:
                self.captions.append(line.strip().lower().split())
            f.close()
        self.length = len(self.captions)

        # image features
        if load_img:
            self.images: np.ndarray = np.load(
                os.path.join(data_path, f"{data_split}_ims.npy")
            )
        else:
            self.images = np.zeros((self.length // caps_per_img, img_dim))

        # each image can have 1 caption or 5 captions
        assert self.images.shape[0] * caps_per_img == self.length
        self.caps_per_img = caps_per_img

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        # image
        img_id = index // self.caps_per_img
        image = torch.tensor(self.images[img_id])
        # caption
        caption = [
            self.vocab(token)
            for token in ["<start>"] + self.captions[index] + ["<end>"]
        ]
        torch_caption = torch.tensor(caption)
        return image, torch_caption, index, img_id

    def __len__(self) -> int:
        return self.length


ImageData = torch.Tensor
CaptionData = torch.Tensor
Id = int
ImgId = int

Batch = Tuple[torch.Tensor, torch.Tensor, List[int], List[Id]]


def collate_fn(data: List[Tuple[ImageData, CaptionData, Id, ImgId]]) -> Batch:
    """ build mini-batch tensors from a list of (image, caption) tuples """
    # sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = (list(_) for _ in zip(*data))
    stacked_images = torch.stack(images, 0)
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
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 2,
    load_img: bool = True,
    img_dim: int = 2048,
) -> "torch.utils.data.DataLoader[Batch]":
    dset = PrecompDataset(data_path, data_split, vocab, load_img, img_dim)
    data_loader = torch.utils.data.DataLoader(
        dataset=dset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return data_loader


def get_train_loaders(
    data_path: str, vocab: "Vocabulary", batch_size: int, workers: int
) -> Tuple["torch.utils.data.DataLoader[Batch]", "torch.utils.data.DataLoader[Batch]"]:
    train_loader = get_precomp_loader(
        data_path, "train", vocab, batch_size, True, workers
    )
    val_loader = get_precomp_loader(data_path, "dev", vocab, batch_size, False, workers)
    return train_loader, val_loader


def get_eval_loader(
    data_path: str,
    split_name: str,
    vocab: "Vocabulary",
    batch_size: int,
    workers: int,
    load_img: bool = False,
    img_dim: int = 2048,
) -> "torch.utils.data.DataLoader[Batch]":
    eval_loader = get_precomp_loader(
        data_path,
        split_name,
        vocab,
        batch_size,
        False,
        workers,
        load_img=load_img,
        img_dim=img_dim,
    )
    return eval_loader
