from typing import Collection, Any, List, Tuple, TypeVar, NewType, TypedDict
from torch import Tensor
from pathlib import Path
from argparse import Namespace, ArgumentParser
import tqdm
import logging
import os
import pickle as pkl
import shutil
import time

import torch

import data
from vocab import Vocabulary
from model import VGNSL
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data
from torch.utils.data import DataLoader


def train(
    opt: Namespace,
    train_loader: "DataLoader",
    model: VGNSL,
    epoch: int,
    val_loader: "DataLoader",
    vocab: Vocabulary,
) -> None:
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    model.train_start()

    end = time.time()
    for i, train_data in enumerate(tqdm.tqdm(train_loader, desc="Training loop")):
        # Always reset to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data, epoch=epoch)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logger.info(
                "Epoch: [{0}][{1}/{2}]\t"
                "{e_log}\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    e_log=str(model.logger),
                )
            )

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            validate(opt, val_loader, model, vocab)


def validate(
    opt: Namespace,
    val_loader: "DataLoader",
    model: VGNSL,
    vocab: Vocabulary,
) -> float:
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs = encode_data(
        model,
        val_loader,
        vocab,
        opt.log_step,
        logger.info,
    )
    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, measure="cosine")
    logger.info(
        "Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr)
    )
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, measure="cosine")
    logger.info(
        "Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanr)
    )
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    return currscore


def save_checkpoint(
    state: Any,
    is_best: bool,
    curr_epoch: int,
    filename: str = "checkpoint.pth.tar",
    prefix: str = "",
) -> None:
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + "model_best.pth.tar")
    shutil.copyfile(prefix + filename, prefix + str(curr_epoch) + ".pth.tar")


def adjust_learning_rate(
    opt: Namespace, optimizer: torch.optim.Optimizer, epoch: int
) -> None:
    """Sets the learning rate to the initial LR
    decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)
) -> List[Tensor]:
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class CheckpointData(TypedDict):
    epoch: int
    model: Any
    best_rsum: float
    opt: Namespace
    Eiters: int


if __name__ == "__main__":
    # hyper parameters
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path", default="../data/mscoco", help="path to datasets"
    )
    parser.add_argument("--margin", default=0.2, type=float, help="rank loss margin")
    parser.add_argument(
        "--num_epochs", default=30, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="size of a training mini-batch"
    )
    parser.add_argument(
        "--word_dim", default=512, type=int, help="dimensionality of the word embedding"
    )
    parser.add_argument(
        "--embed_size",
        default=512,
        type=int,
        help="dimensionality of the joint embedding",
    )
    parser.add_argument(
        "--grad_clip", default=2.0, type=float, help="gradient clipping threshold"
    )
    parser.add_argument(
        "--learning_rate", default=0.0005, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--lr_update",
        default=15,
        type=int,
        help="number of epochs to update the learning rate",
    )
    parser.add_argument(
        "--workers", default=0, type=int, help="number of data loader workers"
    )
    parser.add_argument(
        "--log_step",
        default=10,
        type=int,
        help="number of steps to print and record the log",
    )
    parser.add_argument(
        "--val_step", default=500, type=int, help="number of steps to run validation"
    )
    parser.add_argument(
        "--logger_name", default="../output/", help="path to save the model and log"
    )
    parser.add_argument(
        "--img_dim",
        default=2048,
        type=int,
        help="dimensionality of the image embedding",
    )
    parser.add_argument(
        "--no_imgnorm",
        action="store_true",
        help="Do not normalize the image embeddings.",
    )
    parser.add_argument(
        "--scoring_hidden_dim", type=int, default=128, help="score hidden dim"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="optimizer, can be Adam, SGD, etc.",
    )

    parser.add_argument("--init_embeddings", type=int, default=0)
    parser.add_argument(
        "--init_embeddings_type",
        choices=["override", "partial", "partial-fixed", "subword"],
        default="override",
    )
    parser.add_argument(
        "--init_embeddings_key",
        choices=["glove", "fasttext", "bert"],
        default="override",
        help="If set, we will use one of 'vocab.pkl.glove_embeddings.npy', "
        "'vocab.pkl.fasttext_embeddings.npy', 'vocab.pkl.bert_embeddings.npy' to initalize "
        " embeddings.",
    )
    parser.add_argument("--init_embeddings_partial_dim", type=int, default=0)

    parser.add_argument("--syntax_score", default="conv", choices=["conv", "dynamic"])
    # Currently, syntax is tied with semantics, so we just use word_dim
    # parser.add_argument("--syntax_dim", type=int, default=300)

    # For syntax_score == 'conv'
    parser.add_argument("--syntax_score_hidden", type=int, default=128)
    parser.add_argument("--syntax_score_kernel", type=int, default=5)
    parser.add_argument("--syntax_dropout", type=float, default=0.1)

    # Not used currently
    # parser.add_argument("--syntax_tied_with_semantics", type=int, default=1)
    # parser.add_argument("--syntax_embedding_norm_each_time", type=int, default=1)
    # parser.add_argument("--semantics_embedding_norm_each_time", type=int, default=1)

    parser.add_argument("--vse_reward_alpha", type=float, default=1.0)
    parser.add_argument("--vse_loss_alpha", type=float, default=1.0)

    parser.add_argument(
        "--lambda_hi",
        type=float,
        default=0,
        help="penalization for head-initial inductive bias",
    )
    opt = parser.parse_args()

    # Syntax is tied with semantics
    opt.syntax_dim = opt.word_dim

    # setup logger
    if os.path.exists(opt.logger_name):
        print(f"Warning: the folder {opt.logger_name} exists.")
    os.system("mkdir {:s}".format(opt.logger_name))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = logging.FileHandler(os.path.join(opt.logger_name, "train.log"), "w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False

    subword_suf = ""
    if (opt.init_embeddings_key == "bert") != (
        opt.init_embeddings_type == "subword"
    ):
        raise Exception(
            " --init_embeddings_key bert (must) go along -- --init_embeddings_type subword."
        )
    elif opt.init_embeddings_type == "subword":
        subword_suf = "_subword"

    vocab_filename = f"vocab{subword_suf}.pkl"

    with open(Path(opt.data_path) / vocab_filename, "rb") as fb:
        vocab: Vocabulary = pkl.load(fb)
    opt.vocab_size = len(vocab)

    if opt.init_embeddings:
        opt.vocab_init_embeddings = os.path.join(
            opt.data_path, f"{vocab_filename}.{opt.init_embeddings_key}_embeddings.npy"
        )
    

    # Load data loaders
    train_loader, val_loader = data.get_train_loaders(
        opt.data_path,
        vocab,
        opt.batch_size,
        opt.workers,
        subword=opt.init_embeddings_type == "subword",
    )
    opt.vocab_size = len(vocab)

    # construct the model
    model = VGNSL(opt)

    best_rsum = 0.0

    # Do one test validation run to make sure all is fine
    validate(opt, val_loader, model, vocab)

    for epoch in range(opt.num_epochs):
        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader, vocab)

        # evaluate on validation set using VSE metrics
        rsum = validate(opt, val_loader, model, vocab)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint(
            CheckpointData(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "best_rsum": best_rsum,
                    "opt": opt,
                    "Eiters": model.Eiters,
                }
            ),
            is_best,
            epoch,
            prefix=opt.logger_name + "/",
        )
