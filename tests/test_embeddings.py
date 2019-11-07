import gensim.downloader as api
import numpy as np
import pytest
from zeugma import EmbeddingTransformer
from zeugma.conf import DEFAULT_PRETRAINED_EMBEDDINGS


def test_embedding_instatantiation(sample_corpus_embedding, toy_model_keyed_vectors):
    """Test instantiating an EmbeddingTransformer with an unknown aggregation method."""
    with pytest.raises(ValueError):
        EmbeddingTransformer(
            model=toy_model_keyed_vectors, aggregation="crazy_transform"
        )


def test_corpus_transformation(sample_corpus_embedding, toy_model_keyed_vectors):
    """ Test toy model embedding on a basic corpus """
    embedder = EmbeddingTransformer(model=toy_model_keyed_vectors)
    embeddings = embedder.transform(sample_corpus_embedding)
    assert embeddings.shape[0] == len(sample_corpus_embedding)
    assert np.all(embeddings[1] == embeddings[2])
    with pytest.raises(TypeError):
        embedder.transform([12, "a b"])

    embedder_sum = EmbeddingTransformer(
        model=toy_model_keyed_vectors, aggregation="sum"
    )
    embeddings_sum = embedder_sum.transform(sample_corpus_embedding)
    corpus_words = set(toy_model_keyed_vectors.vocab.keys())
    sample_corpus_overlap = [
        set(sentence.split()) & corpus_words for sentence in sample_corpus_embedding
    ]
    assert np.all(
        embeddings_sum
        == [
            len(sample_corpus_overlap[i]) * embeddings[i]
            for i in range(len(sample_corpus_embedding))
        ]
    )

    embedder_minmax = EmbeddingTransformer(
        model=toy_model_keyed_vectors, aggregation="minmax"
    )
    embeddings_minmax = embedder_minmax.transform(sample_corpus_embedding)
    assert embeddings_minmax.shape[1] == 2 * toy_model_keyed_vectors.vector_size


def test_model_loading(sample_corpus_embedding, toy_model_keyed_vectors):
    """Test model loading exceptions raising."""
    with pytest.raises(TypeError):
        EmbeddingTransformer(model=3)
    with pytest.raises(KeyError):
        EmbeddingTransformer(model="fake_model")


def test_api_model_loading(sample_corpus_embedding):
    """Test embeddings loaded through the Gensim download API."""
    embedder = EmbeddingTransformer(model=list(DEFAULT_PRETRAINED_EMBEDDINGS.keys())[0])
    embeddings = embedder.transform(sample_corpus_embedding)
    assert embeddings.shape[0] == len(sample_corpus_embedding)
    assert np.all(embeddings[1] == embeddings[2])

    embedder = EmbeddingTransformer(model=list(api.info()["models"].keys())[0])
    embeddings = embedder.transform(sample_corpus_embedding)
    assert embeddings.shape[0] == len(sample_corpus_embedding)
    assert np.all(embeddings[1] == embeddings[2])
