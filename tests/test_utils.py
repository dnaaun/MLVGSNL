import torch
import sys
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.utils import SubwordEmbedder
else:
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from utils import SubwordEmbedder


def test_vocab_embedder() -> None:

    vocab_size = 3
    padding_idx = 0
    E = 1
    init_embs = torch.tensor([[99], [1], [2]], dtype=torch.float)
    ember = SubwordEmbedder(
        vocab_size=vocab_size, padding_idx=padding_idx, dim=E, init_embs=init_embs
    )
    print(list(ember.parameters()))

    inp = torch.tensor(
        [
            [[1, 0], [1, 1]],
            [[1, 2], [0, 0]],
        ],
        dtype=torch.long,
    )
    B, L, _ = inp.size()

    res = ember(inp)

    exp_res = torch.tensor([[[1], [1]], [[1.5], [0]]])

    torch.testing.assert_allclose(res, exp_res)
