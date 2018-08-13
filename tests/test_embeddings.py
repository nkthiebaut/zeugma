# -*- coding:utf-8 -*-
"""
Created on the 21/11/17
@author: Nicolas Thiebaut
@email: nicolas@visage.jobs
"""
import pytest
from gensim.sklearn_api import W2VTransformer
from gensim.test.utils import common_texts
import numpy as np
from zeugma import EmbeddingTransformer


class TestEmbeddings(object):
    """ Test embedding transformers """
    @pytest.fixture(scope='class')
    def toy_model_keyed_vectors(self):
        """ Instantiate trainable word2vec vectorizer """
        model = W2VTransformer(size=10, min_count=1, seed=1)
        model.fit(common_texts)
        return model.gensim_model.wv

    def test_corpus_transformation(self, sample_corpus_embedding, toy_model_keyed_vectors):
        """ Test toy model embedding on a basic corpus """
        embedder = EmbeddingTransformer(model=toy_model_keyed_vectors)
        out = embedder.transform(sample_corpus_embedding)
        assert out.shape[0] == len(sample_corpus_embedding)
        assert np.all(out[1] == out[2])
        with pytest.raises(TypeError):
            embedder.transform([12, 'a b'])
        fake_embedder = EmbeddingTransformer(model=toy_model_keyed_vectors, aggregation='crazy_transform')
        with pytest.raises(ValueError):
            fake_embedder.transform(sample_corpus_embedding)

    def test_model_loading(self, sample_corpus_embedding, toy_model_keyed_vectors):
        with pytest.raises(TypeError):
            EmbeddingTransformer(model=3)
        with pytest.raises(KeyError):
            EmbeddingTransformer(model='fake_model')



