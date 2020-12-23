from __future__ import annotations
import pytorch_lightning as pl  # type: ignore
from itertools import count, islice
import csv
from simdjson import Parser  # type: ignore
from dnips.iter.bidict import BiDict
import json
import base64
from functools import cached_property
from random import Random
import pickle as pkl
import math
from typing import (
    Callable,
    Dict,
    Optional,
    Set,
    TypedDict,
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
from .utils import DEBUG


class RegionFeats(NamedTuple):
    img_id: int
    h: int
    w: int

    bxes: np.ndarray[np.float32]
    # (num_boxes, 4)

    feats: np.ndarray[np.float32]
    # num_boxesn, 2048)


class CaptionedEx(NamedTuple):
    id: int
    cap: List[int]
    img: Optional[RegionFeats] = None


def _load_feats_from_tsv(feats_tsv_fps: List[Path]) -> Dict[int, RegionFeats]:
    imgs = {}

    for feats_tsv_fp in feats_tsv_fps:
        with feats_tsv_fp.open() as f:
            reader = csv.reader(f)
            if DEBUG:
                rows = islice(reader, 5000)
            else:
                rows = reader
            for (
                img_id_str,
                w_str,
                h_str,
                num_boxes_str,
                boxes_str,
                feats_str,
            ) in rows:
                img_id = int(img_id_str)

                num_boxes = int(num_boxes_str)
                bxes = np.frombuffer(
                    base64.decodestring(boxes_str.encode()), dtype=np.float32
                ).reshape((num_boxes, -1))
                feats = np.frombuffer(
                    base64.decodestring(feats_str.encode()), dtype=np.float32
                ).reshape((num_boxes, -1))

                imgs[img_id] = RegionFeats(
                    img_id=img_id, w=int(w_str), h=int(h_str), bxes=bxes, feats=feats
                )

    return imgs


def _load_img_ids_from_tsv(feats_tsv_fps: List[Path]) -> Set[int]:
    img_ids = []
    for feats_tsv_fp in feats_tsv_fps:
        with feats_tsv_fp.open() as f:
            reader = csv.reader(f)
            if DEBUG:
                rows = islice(reader, 5000)
            else:
                rows = reader
            for (img_id_str, _, _, _, _, _,) in rows:
                img_ids.append(int(img_id_str))
    return set(img_ids)


MSCOCOCapJsonFmt = TypedDict(
    "MSCOCOCapJsonFmt", {"image_id": int, "id": int, "caption": str}
)


class MSCOCORegionsDataset:
    def __init__(
        self,
        mscoco_jsons: Sequence[MSCOCOCapJsonFmt],
        vocab: BiDict[str, int],
        tokenizer: Callable[[str], List[str]],
        tsv_fps: List[Path],
        load_feats: bool = True,
        unk_tok: str = "<unk>",
        pad_tok: str = "<pad>",
        start_tok: str = "<start>",
        end_tok: str = "<end>",
    ) -> None:

        self._load_feats = load_feats
        if load_feats:
            self._imgs = _load_feats_from_tsv(tsv_fps)
            img_ids = set(self._imgs.keys())
        else:
            img_ids = _load_img_ids_from_tsv(tsv_fps)

        MSCOCOCapInfo = NamedTuple(
            "MSCOCOCapInfo", [("cap", List[int]), ("id", int), ("img_id", int)]
        )
        self._cap_infos: List[MSCOCOCapInfo] = []

        unk_id = vocab[unk_tok]
        for mscoco_json in mscoco_jsons:
            if mscoco_json["image_id"] not in img_ids:
                continue
            cap = [vocab.get(tok, unk_id) for tok in tokenizer(mscoco_json["caption"])]
            self._cap_infos.append(
                MSCOCOCapInfo(cap, mscoco_json["id"], mscoco_json["image_id"])
            )

    def __getitem__(self, i: int) -> CaptionedEx:
        cap_info = self._cap_infos[i]

        img = None
        if self._load_feats:
            img = self._imgs[cap_info.img_id]

        return CaptionedEx(id=cap_info.id, cap=cap_info.cap, img=img)

    def __len__(self) -> int:
        return len(self._cap_infos)


class MSCOCORegionsDataModule(pl.LightningDataModule):
    def __init__(self, data_d: str, batch_size: int) -> None:
        super().__init__()
        self._data_d = Path(data_d)

    def setup(self) -> None:
        with open(self._data_d / "vocab.txt") as f:
            words = [self._preprocess(l.strip()) for l in f]
            words.extend(["<unk>", "<pad>", "<start>", "<end>"])
            self._vocab = BiDict(enumerate(words)).rev

        parser = Parser()
        with open(self._data_d / "captions.json", "b") as fb:
            doc = parser.parse(fb.read())
        self._mscoco_jsons = doc["annotations"]

    def _preprocess(self, word: str) -> str:
        return word.lower()

    @cached_property
    def _train_dataset(self) -> MSCOCORegionsDataset:
        return self._dataset(list(self._data_d.glob("*train*.tsv*")))

    @cached_property
    def _val_dataset(self) -> MSCOCORegionsDataset:
        return self._dataset(list(self._data_d.glob("*val*.tsv*")))

    @cached_property
    def _test_dataset(self) -> MSCOCORegionsDataset:
        return self._dataset(list(self._data_d.glob("*test*.tsv*")), load_feats=False)

    def _dataset(
        self, tsv_fps: List[Path], load_feats: bool = True
    ) -> MSCOCORegionsDataset:
        return MSCOCORegionsDataset(
            self._mscoco_jsons,
            self._vocab,
            self._tokenize,
            tsv_fps,
            load_feats=load_feats,
        )
