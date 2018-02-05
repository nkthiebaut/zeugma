# -*- coding:utf-8 -*-
"""
Created on the 21/11/17
@author: Nicolas Thiebaut
@email: nicolas@visage.jobs
"""

import os
import pytest
import numpy as np
from zeugma import Word2VecTransformer


@pytest.mark.skipif(os.environ.get("TRAVIS") == "true", reason="Travis does'nt work with those tests")
class TestEmbeddings(object):
    """ Test embedding transformers """
    @pytest.fixture(scope='class')
    def word2vec(self):
        """ Instantiate trainable word2vec vectorizer """
        return Word2VecTransformer(trainable=True,
                                   model_path='/tmp/model.bin', min_count=2)

    def test_word2vec(self, sample_corpus, word2vec):
        """ Test word2vec training and transformation on a basic corpus """
        out = word2vec.fit_transform(sample_corpus)
        assert out.shape[0] == len(sample_corpus)
        assert np.all(out[1] == out[2])
