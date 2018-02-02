# -*- coding:utf-8 -*-
"""
Created on the 01/26/2018
@author: Nicolas Thiebaut
@email: nicolas@visage.jobs
"""
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from zeugma.conf import OOV_TAG
from zeugma.logger import PACKAGE_LOGGER as logger


class RareWordsTagger(BaseEstimator, TransformerMixin):
    """ Replace rare words with a token in a corpus (list of strings) """

    def __init__(self, min_count, oov_tag=OOV_TAG):
        self.min_count = min_count
        self.oov_tag = oov_tag
        self.frequencies = defaultdict(int)
        self.kept_words = None

    def fit(self, texts, y=None):
        all_tokens = (token for t in texts for token in t.split())
        for w in all_tokens:
            self.frequencies[w] += 1
        logger.info('Vocabulary size before rare words tagging: ' +
                    str(len(self.frequencies)))
        self.kept_words = {word for word, frequency in self.frequencies.items()
                           if frequency >= self.min_count}
        logger.info('Vocabulary size after rare words tagging: ' +
                    str(len(self.kept_words)))
        return self

    def transform(self, texts):
        texts = [' '.join([w if w in self.kept_words else self.oov_tag
                           for w in t.split()])
                 for t in texts]
        return texts


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        """ Necessary fit method to include transformer in a sklearn.Pipeline"""
        return self

    def transform(self, data_dict):
        """ Return selected items """
        return data_dict[self.key]


class TextStats(FunctionTransformer):
    """Extract features from each document for DictVectorizer"""
    def __init__(self):
        def extract_stats(corpus):
            return [{'length': len(text),
                    'num_sentences': text.count('.')}
                    for text in corpus]
        super().__init__(extract_stats, validate=False)
