import gensim.downloader as api
import numpy as np
import pytest
from gensim.sklearn_api import W2VTransformer
from gensim.test.utils import common_texts


@pytest.fixture(scope="function")
def sample_corpus():
    """ Return a sample corpus in a numpy.array """
    corpus = [
        "Here a first example text",
        "This is a second text with a weird word gwiurgergwggreg",
        "This is a second text with a weird word",
        "Et c'est un troisieme text avec un accent",
    ]
    return np.array(corpus)


@pytest.fixture(scope="function")
def sample_corpus_embedding():
    """ Return a sample corpus in a numpy.array """
    corpus = [
        "human computer",
        "interface gwiurgergwggreg",
        "interface",
        "Et c'est un troisième text avec un accent",
    ]
    return np.array(corpus)


@pytest.fixture(scope="module")
def toy_model_keyed_vectors():
    """ Instantiate trainable word2vec vectorizer """
    model = W2VTransformer(size=10, min_count=1, seed=42)
    model.fit(common_texts)
    return model.gensim_model.wv


@pytest.fixture(autouse=True)
def mock_gensim_api(monkeypatch, toy_model_keyed_vectors):
    """Mock the gensim model loading API: return a small model."""

    def mockreturn(model_name):
        return toy_model_keyed_vectors

    monkeypatch.setattr(api, "load", mockreturn)
