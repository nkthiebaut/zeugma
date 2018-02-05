# -*- coding:utf-8 -*-
"""
Created on the 02/05/2018
@author: Nicolas Thiebaut
@email: nicolas@visage.jobs
"""
import pytest
from sklearn.pipeline import make_pipeline
from zeugma.keras_transformers import TextsToSequences, Padder


class TestTextTransformers(object):
    """ Test Keras transformers """

    @pytest.fixture(scope='class')
    def sequencer(self):
        """ Instantiate trainable word2vec vectorizer """
        return TextsToSequences(num_words=5)

    def test_sequencer(self, sample_corpus, sequencer):
        """ Test text sequencer """
        num_words = sequencer.num_words
        out = sequencer.fit_transform(sample_corpus)
        print(out)
        assert len(out) == len(sample_corpus)
        assert len(out[0]) == 2
        assert max([index for sequence in out for index in sequence]) == num_words - 1

    def test_padder(self, sample_corpus, sequencer):
        """ Test padding uneven sequences """
        padder = Padder(max_length=10)
        out = make_pipeline(sequencer, padder).fit_transform(sample_corpus)
        assert out.shape == (len(sample_corpus), 10)
