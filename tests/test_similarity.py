import numpy as np

from survilai.core.recognition import cosine_similarity


def test_cosine_similarity_identical_vectors() -> None:
    vector = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(vector, vector) == 1.0


def test_cosine_similarity_zero_vector() -> None:
    assert cosine_similarity(np.zeros(3), np.ones(3)) == 0.0
