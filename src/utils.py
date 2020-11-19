import collections
from argparse import Namespace
from functools import reduce

import torch
from torch import Tensor
import numpy as np
import torch.nn as nn
from typing import Sequence, Tuple, Union, List, TYPE_CHECKING, cast, TypeVar, Optional

if TYPE_CHECKING:
    from vocab import Vocabulary


class EmbeddingCombiner(nn.Module):
    def __init__(self, *embeddings: nn.Module):
        super().__init__()
        self.embeddings = nn.ModuleList(embeddings)

    def forward(self, input: Tensor) -> Tensor:
        return torch.cat([e(input) for e in self.embeddings], dim=-1)


class SubwordEmbedder(nn.Module):
    """Pools over subwords. Look at forward() for docs.

    THIS MODULE DEPENDS ON THE PADDING EMBEDDING BEING ALL ZEROS. IT WILL SET THIS IN __init__.
    PLEASE BE CAREFUL NOT TO DO SOMETHING LIKE:

        >>> subword_embedder._subword_embs.weight.data.copy_(fasttext_embs)

    Instead, pass `fasttext_embs` as a param to __init__

    """

    def __init__(
        self, vocab_size: int, dim: int, padding_idx: int = 0, init_embs: Tensor = None
    ) -> None:
        super().__init__()
        self._padding_idx = padding_idx
        self._subword_embs = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        # if init_embs is not None:
        # with torch.no_grad():
        # self._subword_embs.weight.copy_(init_embs.detach())

        self._zero: nn.Parameter
        self.register_buffer(
            "_zero", nn.Parameter(torch.tensor(0.0, requires_grad=False))
        )

        # Zero out the padding idx
        with torch.no_grad():
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
                emb.weight.detach().copy_(init_embeddings)
            else:
                assert opt.init_embeddings_type == "partial"
                emb.weight.detach()[:, : init_embeddings.size(1)] = init_embeddings
    elif opt.init_embeddings_type == "partial-fixed":
        partial_dim = opt.init_embeddings_partial_dim
        emb1 = nn.Embedding(vocab_size, partial_dim, padding_idx=0)
        emb2 = nn.Embedding(vocab_size, dim - partial_dim, padding_idx=0)

        if init_embeddings is not None:
            emb1.weight.data.detach().copy_(init_embeddings)
        emb1.weight.requires_grad_(False)

        emb = EmbeddingCombiner(emb1, emb2)
    elif opt.init_embeddings_type == "subword":
        emb = SubwordEmbedder(vocab_size, dim, init_embs=init_embeddings)
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


def add_dim(tensor: Tensor, dim: int, size: int) -> Tensor:
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


def generate_tree(
    captions: Tensor,
    tree_indices: Tensor,
    pos: int,
    vocab: "Vocabulary",
    subword: bool,
    pad_word="<pad>",
    pad_word_id: int = 0,
):
    if subword:

        def get_word_func(__word_ids: Tensor) -> Optional[str]:
            subwords = [
                vocab.idx2word[int(word_id)]
                for word_id in __word_ids
                if word_id != pad_word_id
            ]
            if subwords:
                return "|".join(subwords)
            return None

    else:

        def get_word_func(__word_id: Tensor) -> Optional[str]:
            if __word_id == pad_word_id:
                return None
            return vocab.idx2word[int(__word_id)]

    words = []
    for word_ids in captions[pos]:
        word = get_word_func(word_ids)
        if word is None:
            continue
        word = {"(": "-LBR-", ")": "-RBR-"}.get(word, word)
        words.append(word)

    idx = 0
    while len(words) > 1:
        p = cast(int, tree_indices[idx][pos])
        words = (
            words[:p]
            + ["( {:s} {:s} )".format(words[p], words[p + 1])]
            + words[p + 2 :]
        )
        idx += 1
    return words[0]


def sequence_mask(
    # (B,)
    sequence_length: Tensor,
    max_length: Tensor = None,
) -> Tensor:
    """Returns a 2D binary tensor that can be used for masking padding positions."""
    if max_length is None:
        max_length = sequence_length.data.max()
    batch_size = sequence_length.size(0)

    # (L,)
    seq_range = torch.arange(0, max_length).long()  # type: ignore # Using a 0dim tensor

    # (B, L)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length)

    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()

    # (B, L)
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)

    # (B, L)
    return seq_range_expand < seq_length_expand


def index_one_hot_ellipsis(
    # (B, *)
    tensor: Tensor,
    # Always 1 actually.
    dim: int,
    # (B,)
    index: Tensor,
) -> Tensor:
    tensor_shape = tensor.size()

    # (B, *, 1)
    tensor = tensor.view(
        prod(tensor_shape[:dim]), tensor_shape[dim], prod(tensor_shape[dim + 1 :])
    )
    assert tensor.size(0) == index.size(0)

    # (B,1,1)
    index = index.unsqueeze(-1).unsqueeze(-1)

    # (B,1,1)
    index = index.expand(tensor.size(0), 1, tensor.size(2))

    # (B,1,1)
    tensor = tensor.gather(1, index)

    # (B, *)
    return tensor.view(tensor_shape[:dim] + tensor_shape[dim + 1 :])


def index_mask(indices: Tensor, max_length: int) -> Tensor:
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


_Num = TypeVar("_Num", int, float)


def prod(values: Sequence[_Num], default: int = 1) -> _Num:
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
