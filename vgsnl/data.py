from __future__ import annotations
import allennlp
import sys

from allennlp.data.fields.field import Field

from allennlp.data.vocabulary import Vocabulary
import pytorch_lightning as pl  # type: ignore
from allennlp.data.fields import (
    TensorField,
    LabelField,
    ListField,
    TextField,
    MetadataField,
)
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.common.file_utils import TensorCache
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from itertools import count, islice
import csv
# from simdjson import Parser  # type: ignore
import json
from dnips.iter.bidict import BiDict
import json
import base64
from functools import cached_property
from random import Random
from logging import Logger, basicConfig
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

basicConfig()
logger = Logger(__name__)


class RegionFeats(NamedTuple):
    height: int
    width: int

    boxes: np.ndarray[np.float32]
    # (num_boxes, 4)

    features: np.ndarray[np.float32]
    # num_boxesn, 2048)


class CaptionedEx(NamedTuple):
    id: int
    cap: List[int]
    img: Optional[RegionFeats] = None


MSCOCOCapJsonFmt = TypedDict(
    "MSCOCOCapJsonFmt", {"image_id": int, "id": int, "caption": str}
)


class MSCOCORegionsReader(DatasetReader):
    def __init__(
        self,
        data_dir: Path,
        tokenizer: Tokenizer = None,
        caption_indexers: Dict[str, TokenIndexer] = None,
        load_feats: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._data_dir = data_dir
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = caption_indexers or {"tokens": SingleIdTokenIndexer()}
        self._load_feats = load_feats

    def _read(self, img_feat_tsvs: List[str]) -> Iterable[Instance]:
        ## Parse
        if self._load_feats:
            imgs = self._load_feats_from_tsv([Path(i) for i in img_feat_tsvs])
            image_ids = set(imgs.keys())
        else:
            image_ids = self._load_image_ids_from_tsv([Path(i) for i in img_feat_tsvs])

        # The Karpathy split spans across the original train and val splits.
        # accordingly,  read all the json files in the data dir.
        # parser = Parser()
        for json_file in self._data_dir.glob("*.json"):
            with open(json_file) as f:
                # caption_infos = parser.parse(str(json_file))
                annotations = json.load(f)["annotations"]

                for annotation in annotations:
                    if annotation["image_id"] not in image_ids:
                        continue

                    image_id = annotation["image_id"]
                    if self._load_feats:
                        assert imgs  # type: ignore
                        image_attrs = imgs[image_id]._asdict()
                    else:
                        image_attrs = {}

                    instance = self.text_to_instance(
                        caption=annotation["caption"],
                        caption_id=annotation["id"],
                        image_id=image_id,
                        **image_attrs
                    )

                    yield instance


    def text_to_instance(
        self,
        caption: str,
        caption_id: int,
        image_id: int,
        width: int = None,
        height: int = None,
        boxes: np.ndarray[np.float32] = None,
        features: np.ndarray[np.float32] = None,
    ) -> Instance:
        fields: Dict[str, Field] = {
            "caption": TextField(self._tokenizer.tokenize(caption)),
            "metadata": MetadataField({"id": caption_id, "image_id": image_id}),
        }

        if boxes is not None or width or height or features is not None:

            if boxes is None or features is None or width is None or height is None:
                raise Exception("Either provide all image attributes, or none of them.")

            fields["boxes"] = TensorField(boxes)
            fields["features"] = TensorField(features)
            fields["width"] = TensorField(torch.tensor(width))
            fields["height"] = TensorField(torch.tensor(height))

        return Instance(fields)

    @staticmethod
    def _load_feats_from_tsv(feats_tsv_fps: List[Path]) -> Dict[int, RegionFeats]:

        imgs = {}
        csv.field_size_limit(sys.maxsize)
        for feats_tsv_fp in feats_tsv_fps:
            with feats_tsv_fp.open() as f:
                reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
                if DEBUG:
                    rows = islice(reader, 5000)
                else:
                    rows = reader
                for (
                    image_id_str,
                    w_str,
                    h_str,
                    num_boxes_str,
                    boxes_str,
                    feats_str,
                ) in rows:
                    image_id = int(image_id_str)

                    num_boxes = int(num_boxes_str)
                    boxes = np.frombuffer(
                        base64.decodebytes(boxes_str.encode()), dtype=np.float32
                    ).reshape((num_boxes, -1))
                    features = np.frombuffer(
                        base64.decodebytes(feats_str.encode()), dtype=np.float32
                    ).reshape((num_boxes, -1))

                    # Copy, since a torch.from_numpy call later doens't like these 
                    # read only numpy arrays that come np.frombuffer
                    boxes = np.copy(boxes)
                    features = np.copy(features)

                    imgs[image_id] = RegionFeats(
                        width=int(w_str),
                        height=int(h_str),
                        boxes=boxes,
                        features=features,
                    )

        return imgs

    @staticmethod
    def _load_image_ids_from_tsv(feats_tsv_fps: List[Path]) -> Set[int]:
        image_ids = []
        csv.field_size_limit(sys.maxsize)
        for feats_tsv_fp in feats_tsv_fps:
            with feats_tsv_fp.open() as f:
                reader = csv.reader(f)
                if DEBUG:
                    rows = islice(reader, 5000)
                else:
                    rows = reader
                for (image_id_str, _, _, _, _, _,) in rows:
                    image_ids.append(int(image_id_str))
        return set(image_ids)
