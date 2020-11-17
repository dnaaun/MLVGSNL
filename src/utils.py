import collections
from argparse import Namespace
from functools import reduce

import torch
from torch import Tensor
import numpy as np
import torch.nn as nn
from typing import Sequence, Tuple, Union, List, TYPE_CHECKING


if TYPE_CHECKING:
    ModuleBase = nn.Module[Tensor]
else:
    ModuleBase = nn.Module


class EmbeddingCombiner(ModuleBase):
    def __init__(self, *embeddings: ModuleBase):
        super().__init__()
        self.embeddings = nn.ModuleList(embeddings)

    def forward(self, input: Tensor) -> Tensor:
        return torch.cat([e(input) for e in self.embeddings], dim=-1)


class SubwordEmbedder(ModuleBase):
    """Pools over subwords. Look at forward() for docs.

    THIS MODULE DEPENDS ON THE PADDING EMBEDDING BEING ALL ZEROS. IT WILL SET THIS IN __init__.
    PLEASE BE CAREFUL NOT TO DO SOMETHING LIKE:

        >>> subword_embedder._subword_embs.weight.data.copy_(fasttext_embs)

    Instead, pass `fasttext_embs` as a param to __init__

    """

    def __init__(
        self, vocab_size: int, dim: int, padding_idx=0, init_embs: Tensor = None
    ) -> None:
        super().__init__()
        self._padding_idx = padding_idx
        self._subword_embs = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        if init_embs is not None:
            self._subword_embs.weight.data.copy_(init_embs)

        # Initialize zero vector here so that it gets moved to GPU automatically if necessary
        self._zero = torch.tensor(0.0, requires_grad=False)

        # Zero out the padding idx
        self._subword_embs.weight[padding_idx] = self._zero

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Args:
            token_ids: tensor of shape (B, L, N)
                where B is batch size
                      L is sequence length
                      N is maximum number of subwords per word

                If a word has less than N subwords, the remaining ids must be padding_idx.
        Returns:
            a tensor of shape (B, L, dim)
                where dim is self._subword_embs.dim
        """
        # (B, L, N, dim)
        not_pooled = self._subword_embs(token_ids)
        # (B, L)
        num_subwords = (token_ids != self._padding_idx).sum(dim=-1)

        # (B, L, dim)
        summed = not_pooled.sum(dim=-2)
        # (B, L, dim)
        pooled = torch.where(
            (num_subwords == self._zero).unsqueeze(-1),  # condition
            self._zero,  # if true
            summed / num_subwords.unsqueeze(-1),  # if false
        )

        return pooled


def tree2list(tokens):
    tree = list()
    list_stack = list()
    list_stack.append(tree)
    stack_top = tree
    for token in tokens:
        if token == "(":
            new_span = []
            stack_top.append(new_span)
            list_stack.append(new_span)
            stack_top = new_span
        elif token == ")":
            list_stack = list_stack[:-1]
            if len(list_stack) != 0:
                stack_top = list_stack[-1]
        else:
            stack_top.append(token)
    return tree


def treelist2dict(tree, d):
    if type(tree) is str:
        return tree
    span_reprs = [treelist2dict(s, d) for s in tree]
    d[" ".join(span_reprs)] = tree
    return " ".join(span_reprs)


def tree2str(tree):
    if type(tree) is str:
        return tree
    items = [tree2str(item) for item in tree]
    return "( " + " ".join(items) + " )"


def make_embeddings(opt: Namespace, vocab_size: int, dim: int) -> "nn.Module[Tensor]":
    init_embeddings = None
    if hasattr(opt, "vocab_init_embeddings"):
        init_embeddings = torch.from_numpy(np.load(opt.vocab_init_embeddings))

    emb = None
    if opt.init_embeddings_type in ("override", "partial"):
        emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        if init_embeddings is not None:
            if opt.init_embeddings_type == "override":
                emb.weight.data.copy_(init_embeddings)
            else:
                assert opt.init_embeddings_type == "partial"
                emb.weight.data[:, : init_embeddings.size(1)] = init_embeddings
    elif opt.init_embeddings_type == "partial-fixed":
        partial_dim = opt.init_embeddings_partial_dim
        emb1 = nn.Embedding(vocab_size, partial_dim, padding_idx=0)
        emb2 = nn.Embedding(vocab_size, dim - partial_dim, padding_idx=0)

        if init_embeddings is not None:
            emb1.weight.data.copy_(init_embeddings)
        emb1.weight.requires_grad_(False)

        emb = EmbeddingCombiner(emb1, emb2)
    elif opt.init_embeddings_type == "bert":
        emb = SubwordEmbedder(vocab_size, dim, init_embeddings)
    else:
        raise NotImplementedError()

    return emb


def concat_shape(*shapes: Union[int, Sequence[int]]) -> Tuple[int, ...]:
    output: List[int] = []
    for s in shapes:
        if isinstance(s, collections.Sequence):
            output.extend(s)
        else:
            output.append(int(s))
    return tuple(output)


def broadcast(tensor: Tensor, dim: int, size: int) -> Tensor:
    if dim < 0:
        dim += tensor.dim()
    assert tensor.size(dim) == 1
    shape = tensor.size()
    return tensor.expand(concat_shape(shape[:dim], size, shape[dim + 1 :]))


def add_dim(tensor, dim: int, size: int) -> Tensor:
    return broadcast(tensor.unsqueeze(dim), dim, size)


def cosine_sim(im: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between all the image and sentence pairs

    Args:
        im: (N,M)
        s: (P,M)

    Returns:
        out: (N,M)
    """
    return im.mm(s.t())


def l2norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(2, dim=dim, keepdim=True).clamp(min=1e-6)  # type: ignore[no-untyped-call,no-any-return]


def generate_tree(captions, tree_indices, pos, vocab, pad_word="<pad>"):
    words = list(
        filter(
            lambda x: x != pad_word,
            [
                {"(": "**LP**", ")": "**RP**"}.get(
                    vocab.idx2word[int(word)], vocab.idx2word[int(word)]
                )
                for word in captions[pos]
            ],
        )
    )
    idx = 0
    while len(words) > 1:
        p = tree_indices[idx][pos]
        words = (
            words[:p]
            + ["( {:s} {:s} )".format(words[p], words[p + 1])]
            + words[p + 2 :]
        )
        idx += 1
    return words[0]


def sequence_mask(sequence_length, max_length=None):
    if max_length is None:
        max_length = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_length).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length)
    seq_range_expand = seq_range_expand
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand


def index_one_hot_ellipsis(tensor, dim, index):
    tensor_shape = tensor.size()
    tensor = tensor.view(
        prod(tensor_shape[:dim]), tensor_shape[dim], prod(tensor_shape[dim + 1 :])
    )
    assert tensor.size(0) == index.size(0)
    index = index.unsqueeze(-1).unsqueeze(-1)
    index = index.expand(tensor.size(0), 1, tensor.size(2))
    tensor = tensor.gather(1, index)
    return tensor.view(tensor_shape[:dim] + tensor_shape[dim + 1 :])


def index_mask(indices, max_length):
    batch_size = indices.size(0)
    seq_range = torch.arange(0, max_length).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length)
    seq_range_expand = seq_range_expand
    if indices.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    indices_expand = indices.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand == indices_expand


def index_range_ellipsis(x, a, b, dim=1, padding_zero=True):
    assert dim == 1

    batch_size, seq_length = x.size()[:2]
    seg_lengths = b - a
    max_seg_length = seg_lengths.max().item()

    mask = length2mask(seg_lengths, max_seg_length)

    # indices values: [[0, 1, 0, 0, ...], [1, 2, 3, 0, ...], ...]
    base = torch.arange(max_seg_length)
    if torch.cuda.is_available():
        base = base.cuda()
    indices = add_dim(base, 0, batch_size) + a.unsqueeze(-1)
    indices = indices * mask.long()  # shape: [batch_size, max_seg_length]

    # batch_indices values: [[0, 0, 0...], [1, 1, 1, ...], ...]
    base = torch.arange(batch_size)
    if torch.cuda.is_available():
        base = base.cuda()
    batch_indices = add_dim(
        base, 1, max_seg_length
    )  # shape: [batch_size, max_seg_length]

    flattened_x = x.reshape(concat_shape(-1, x.size()[2:]))
    flattened_indices = (indices + batch_indices * seq_length).reshape(-1)
    output = flattened_x[flattened_indices].reshape(
        concat_shape(batch_size, max_seg_length, x.size()[2:])
    )

    if padding_zero:
        output = output * add_dim_as_except(mask.type_as(output), output, 0, 1)

    return output


def add_dim_as_except(tensor, target, *excepts):
    assert len(excepts) == tensor.dim()
    tensor = tensor.clone()
    excepts = [e + target.dim() if e < 0 else e for e in excepts]
    for i in range(target.dim()):
        if i not in excepts:
            tensor.unsqueeze_(i)
    return tensor


def length2mask(lengths, max_length):
    rng = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)
    lengths = lengths.unsqueeze(-1)
    rng = add_dim_as_except(rng, lengths, -1)
    mask = rng < lengths
    return mask.float()


def prod(values, default=1):
    if len(values) == 0:
        return default
    return reduce(lambda x, y: x * y, values)


def clean_tree(sentence, remove_tag_set={"<start>", "<end>", "<pad>"}):
    for tag in remove_tag_set:
        sentence = sentence.replace(tag, " ")
    items = sentence.split()
    stack = list()
    for item in items:
        if item != ")":
            stack.append(item)
        else:
            pos = -1
            while stack[pos] != "(":
                pos -= 1
            if pos == -2:
                stack = stack[:-2] + [stack[-1]]
            else:
                stack = stack[:pos] + [" ".join(["("] + stack[pos + 1 :] + [")"])]
    assert len(stack) == 1
    return stack[0]
