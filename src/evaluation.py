from __future__ import annotations
import os
from pathlib import Path
import pickle

from typing import (
    TYPE_CHECKING,
    List,
    Dict,
    Callable,
    Any,
    Tuple,
    overload,
    Union,
    cast,
)
from typing_extensions import Literal
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

from model import VGNSL
from data import get_eval_loader, SingleDataset, Example, Batch
from vocab import Vocabulary
from utils import generate_tree, clean_tree

if TYPE_CHECKING:
    from train import CheckpointData


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 0) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (0.0001 + self.count)

    def __str__(self) -> str:
        """String representation for logging"""
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return "%.4f (%.4f)" % (self.val, self.avg)


class LogCollector:
    """A collection of logging objects that can change from train to val"""

    def __init__(self) -> None:
        # to keep the order of logged variables deterministic
        self.meters: Dict[str, AverageMeter] = OrderedDict()

    def update(self, k: str, v: float, n: int = 0) -> None:
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self) -> str:
        """Concatenate the meters in one log line"""
        s = ""
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += "  "
            s += k + " " + str(v)
        return s


def encode_data(
    model: VGNSL,
    data_loader: DataLoader[Example, Batch],
    vocab: Vocabulary,
    log_step: int = 10,
    logging: Callable[[Any], None] = print,
    stage: Literal["dev", "test", "train"] = "dev",
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode all images and captions loadable by `data_loader`"""
    batch_time = AverageMeter()
    val_logger = LogCollector()
    # switch to evaluate modPrecompDsetBase
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    logged = False
    for i, (lang, (images, captions, lengths, ids)) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger
        tensor_lengths = torch.Tensor(lengths).long()
        if torch.cuda.is_available():
            tensor_lengths = tensor_lengths.cuda()

        # compute the embeddings
        model_output = model.forward_emb(
            lang, images, captions, tensor_lengths, volatile=True
        )
        (
            img_emb,
            cap_span_features,
            left_span_features,
            right_span_features,
            word_embs,
            tree_indices,
            all_probs,
            span_bounds,
        ) = model_output[:8]

        # output sampled trees
        if (not logged) or (stage == "test"):
            logged = True
            sample_num = 5
            for j in range(sample_num):
                logging(
                    generate_tree(
                        captions,
                        tree_indices,
                        j,
                        vocab,
                        subword=cast(SingleDataset, data_loader.dataset).subword,
                    )
                )

        cap_emb = torch.cat(
            [
                cap_span_features[l - 2][i].reshape(1, -1)
                for i, l in enumerate(tensor_lengths)
            ],
            dim=0,
        )

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging(
                "Test: [{0}/{1}]\t"
                "{e_log}\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t".format(
                    i, len(data_loader), batch_time=batch_time, e_log=str(model.logger)
                )
            )
        del images, captions

    assert img_embs is not None and cap_embs is not None
    return img_embs, cap_embs


@overload
def i2t(
    images: np.ndarray,
    captions: np.ndarray,
    *,
    return_ranks: Literal[False] = False,
    npts: int = None,
    measure: Literal["cosine"] = "cosine",
) -> Tuple[float, float, float, float, float]:
    ...


@overload
def i2t(
    images: np.ndarray,
    captions: np.ndarray,
    *,
    return_ranks: Literal[True],
    npts: int = None,
    measure: Literal["cosine"] = "cosine",
) -> Tuple[Tuple[float, float, float, float, float], Tuple[np.ndarray, np.ndarray]]:
    ...


def i2t(
    images: np.ndarray,
    captions: np.ndarray,
    npts: int = None,
    measure: Literal["cosine"] = "cosine",
    return_ranks: bool = False,
) -> Union[
    Tuple[Tuple[float, float, float, float, float], Tuple[np.ndarray, np.ndarray]],
    Tuple[float, float, float, float, float],
]:
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
        # print(npts)
    index_list = []

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        d = np.dot(im, captions.T).flatten()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp: int = np.where(inds == i)[0][0]  # type: ignore
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr: float = np.floor(np.median(ranks)) + 1  # type: ignore
    meanr: float = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


@overload
def t2i(
    images: np.ndarray,
    captions: np.ndarray,
    *,
    return_ranks: Literal[False] = False,
    npts: int = None,
    measure: Literal["cosine"] = "cosine",
) -> Tuple[float, float, float, float, float]:
    ...


@overload
def t2i(
    images: np.ndarray,
    captions: np.ndarray,
    *,
    return_ranks: Literal[True],
    npts: int = None,
    measure: Literal["cosine"] = "cosine",
) -> Tuple[Tuple[float, float, float, float, float], Tuple[np.ndarray, np.ndarray]]:
    ...


def t2i(
    images: np.ndarray,
    captions: np.ndarray,
    *,
    return_ranks: bool = False,
    npts: int = None,
    measure: Literal["cosine"] = "cosine",
) -> Union[
    Tuple[Tuple[float, float, float, float, float], Tuple[np.ndarray, np.ndarray]],
    Tuple[float, float, float, float, float],
]:
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
        # print(npts)
    ims = np.array([images[i] for i in range(0, len(images), 5)])

    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index : 5 * index + 5]

        # compute scores
        d = np.dot(queries, ims.T)
        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]  # type: ignore

    # compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr: float = np.floor(np.median(ranks)) + 1  # type: ignore
    meanr: float = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def test_trees(
    model_path: str,
    test_data_path: Path = None,
) -> Tuple[List[str], List[str]]:
    """ use the trained model to generate parse trees for text """
    # load model and options
    checkpoint: "CheckpointData" = torch.load(model_path, map_location="cpu")
    opt = checkpoint["opt"]

    # load vocabulary used by the model
    vocab_file = "vocab.pkl"
    if opt.init_embeddings_type == "subword":
        vocab_file = "vocab_subword.pkl"
    vocab = pickle.load(open(os.path.join(opt.data_path, vocab_file), "rb"))
    opt.vocab_size = len(vocab)

    # construct model
    model = VGNSL(opt)

    # load model state
    model.load_state_dict(checkpoint["model"])

    if test_data_path is None:
        test_data_path = Path(opt.data_path)

    print("Loading dataset..")
    data_loader = get_eval_loader(
        data_path=str(test_data_path),
        split_name="test",
        vocab=vocab,
        batch_size=opt.batch_size,
        workers=opt.workers,
        subword=opt.init_embeddings_type == "subword",
        load_img=False,
        img_dim=opt.img_dim,
    )

    cap_embs = None
    logged = False
    trees: List[str] = list()
    for i, (lang, (images, captions, lengths, ids)) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = print  # type: ignore # TODO: This actually looks unsafe. But "if it aint broke, ..."
        tensor_lengths = torch.Tensor(lengths).long()
        if torch.cuda.is_available():
            tensor_lengths = tensor_lengths.cuda()

        # compute the embeddings
        model_output = model.forward_emb(
            lang, images, captions, tensor_lengths, volatile=True
        )
        (
            img_emb,
            cap_span_features,
            left_span_features,
            right_span_features,
            word_embs,
            tree_indices,
            all_probs,
            span_bounds,
        ) = model_output[:8]

        candidate_trees = list()
        for j in range(len(ids)):
            candidate_trees.append(
                generate_tree(
                    captions,
                    tree_indices,
                    j,
                    vocab,
                    cast(SingleDataset, data_loader.dataset).subword,
                )
            )
        appended_trees: List[str] = ["" for _ in range(len(ids))]
        for j in range(len(ids)):
            appended_trees[ids[j] - min(ids)] = clean_tree(candidate_trees[j])
        trees.extend(appended_trees)
        cap_emb = torch.cat(
            [
                cap_span_features[l - 2][i].reshape(1, -1)
                for i, l in enumerate(tensor_lengths)
            ],
            dim=0,
        )
        del images, captions, img_emb, cap_emb

    ground_truth = [
        line.strip()
        for line in open(os.path.join(opt.data_path, "test_ground-truth.txt"))
    ]
    return trees, ground_truth
