# -*- coding:utf-8 -*-
"""
Created on the 02/05/2018
@author: Nicolas Thiebaut
@email: nicolas@visage.jobs
"""
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.base import BaseEstimator
from sklearn.pipeline import TransformerMixin


class TextsToSequences(Tokenizer, BaseEstimator, TransformerMixin):
    """ Sklearn transformer to convert texts to indices list

    Example
    -------
    >>> from zeugma import TextsToSequences
    >>> sequencer = TextsToSequences()
    >>> sequencer.fit_transform(["the cute cat", "the dog"])
    [[1, 2, 3], [1, 4]]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, texts, y=None):
        self.fit_on_texts(texts)
        return self

    def transform(self, texts, y=None):
        return np.array(self.texts_to_sequences(texts))


class Padder(BaseEstimator, TransformerMixin):
    """ Pad and crop uneven lists to the same length.
    Only the end of lists longer than the max_length attribute are
    kept, and lists shorter than max_length are left-padded with zeros

    Attributes
    ----------
    max_length: int
        sizes of sequences after padding
    max_index: int
        maximum index known by the Padder, if a higher index is met during
        transform it is transformed to a 0
    """

    def __init__(self, max_length=500):
        self.max_length = max_length
        self.max_index = None

    def fit(self, X, y=None):
        self.max_index = pad_sequences(X, maxlen=self.max_length).max()
        return self

    def transform(self, X, y=None):
        X = pad_sequences(X, maxlen=self.max_length)
        X[X > self.max_index] = 0
        return X


if __name__ == "__main__":
    import doctest

    doctest.testmod()
