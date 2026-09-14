from core.embedder import embed_texts


def test_embed_texts_returns_one_vector_per_input():
    vectors = embed_texts(["hello world", "the quick brown fox"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 384  # all-MiniLM-L6-v2 output dimension
    assert len(vectors[1]) == 384


def test_embed_texts_is_deterministic():
    a = embed_texts(["the payment is due on the first of the month"])
    b = embed_texts(["the payment is due on the first of the month"])

    assert a[0] == b[0]


def test_similar_texts_are_closer_than_unrelated_texts():
    import numpy as np

    anchor, similar, unrelated = embed_texts(
        [
            "The payment due date is the first of every month.",
            "Payments must be made by the 1st of each month.",
            "The cafeteria serves lunch between noon and 2pm.",
        ]
    )

    def cosine(a, b):
        a, b = np.array(a), np.array(b)
        return a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

    assert cosine(anchor, similar) > cosine(anchor, unrelated)
