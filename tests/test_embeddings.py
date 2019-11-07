import numpy as np
import pytest
from gensim.sklearn_api import W2VTransformer
from gensim.test.utils import common_texts
from zeugma import EmbeddingTransformer


class TestEmbeddings(object):
    """ Test embedding transformers """

    @pytest.fixture(scope="class")
    def toy_model_keyed_vectors(self):
        """ Instantiate trainable word2vec vectorizer """
        model = W2VTransformer(size=10, min_count=1, seed=42)
        model.fit(common_texts)
        return model.gensim_model.wv

    def test_embedding_instatantiation(
        self, sample_corpus_embedding, toy_model_keyed_vectors
    ):
        with pytest.raises(ValueError):
            EmbeddingTransformer(
                model=toy_model_keyed_vectors, aggregation="crazy_transform"
            )

    def test_corpus_transformation(
        self, sample_corpus_embedding, toy_model_keyed_vectors
    ):
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
        corpus_words = set(toy_model_keyed_vectors.wv.vocab.keys())
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

    def test_model_loading(self, sample_corpus_embedding, toy_model_keyed_vectors):
        with pytest.raises(TypeError):
            EmbeddingTransformer(model=3)
        with pytest.raises(KeyError):
            EmbeddingTransformer(model="fake_model")
