from allennlp.data import token_indexers

from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from vgsnl.utils import make_embeddings
from vgsnl.data import MSCOCORegionsReader
from allennlp.data import Vocabulary
from pathlib import Path
from allennlp.data.token_indexers import SingleIdTokenIndexer
import pytest

TEST_DATA_D = Path(__file__).parent / "test_data"


@pytest.mark.skipif(
    not TEST_DATA_D.exists(), reason=f"{str(TEST_DATA_D)} doesn't exist."
)
def test_regions_reader() -> None:

    tokenizer = SpacyTokenizer()
    caption_indexer = SingleIdTokenIndexer(namespace="caption")
    vocab = Vocabulary()

    data_dir = TEST_DATA_D /  "test_mscoco_karpathy"
    reader = MSCOCORegionsReader(
        data_dir=data_dir,
        tokenizer=tokenizer,
        caption_indexers={"caption": caption_indexer},
        load_feats=True,
    )

    dataset = reader.read([data_dir / "val.tsv"])

    for instance in dataset:

        assert len(instance["boxes"]) == len(instance["features"])
