import pytest
import data
import numpy as np
from sentence_transformers import SentenceTransformer


SHORT_TEXT = "Short text"
LONG_TEXT = "Short text, many repeats" * 200


@pytest.fixture(scope="session")
def model() -> SentenceTransformer:
    res = SentenceTransformer('BAAI/bge-small-en-v1.5', device="cpu")
    return res


def test_load_dataset():
    texts, labels = data.load_dataset()
    assert len(texts) == 50000


def test_long_text_indices(model: SentenceTransformer):
    texts = [SHORT_TEXT, LONG_TEXT]
    assert data.get_long_texts_indices(texts, model.max_seq_length, model.tokenizer) == [1]


@pytest.mark.parametrize("text, expected", [
    (SHORT_TEXT, True),
    (LONG_TEXT, False),
])
def test_get_long_text_embedding(model: SentenceTransformer, text: str, expected: bool):
    emb1 = model.encode(text)
    emb2 = data.get_long_text_embedding(text, model)
    assert emb1.shape == emb2.shape
    assert np.array_equal(emb1, emb2) == expected
