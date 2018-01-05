# -*- coding:utf-8 -*-
"""
Created on the 05/01/18
@author: Nicolas Thiebaut
@email: nkthiebaut@gmail.com
"""

from abc import ABCMeta, abstractmethod
import gzip
from multiprocessing import cpu_count
import os
import shutil
import urllib
import zipfile

import fastText
from gensim.models import KeyedVectors, Word2Vec
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


FASTTEXT_URL = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/" +\
    "wiki.simple.zip" # English light version: 2.4 GB
# FASTTEXT_URL = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/" +\
#   "wiki.en.zip"  # English large version: 9.6GB



class EmbeddingTransformer(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    """ Abstract text vectorizer class """

    def __init__(self, model_path=None, trainable=False, **kwargs):
        if trainable:
            for method in ['train', 'save']:
                if not hasattr(self, method):
                    raise NotImplementedError('trainable is set to True but ' +\
                            self.__class__.__name__ + ' does not implement ' + \
                                         method + ' method.')
        elif model_path is None:
            raise ValueError(self.__class__.__name__ + ' constructor needs ' + \
                             'least a model_path, or "trainable" set to True')
        elif not hasattr(self, 'load_pretrained_model'):
            raise NotImplementedError(self.__class__.__name__ + \
                                      ' does not support pretrained models.')
        else:
            if not os.path.exists(model_path):
                raise FileNotFoundError(self.__class__.__name__ +
                                        ' model file not found')
            self.model = self.load_pretrained_model()
        self.trainable = trainable
        self.model_path = model_path
        self.params = dict(kwargs)

    @abstractmethod
    def transform_sentence(self, text):
        """ Child classes have to implement this method that will be used
        to transform corpora """
        pass

    def fit(self, x, y=None):
        """ Has to define fit method to conform scikit-learn Transformer
        definition and integrate a sklearn.Pipeline object """
        if self.trainable:
            self.train(x)
            self.save()
        return self

    def transform(self, texts):
        """ Transform corpus from single text transformation method """
        if hasattr(self, 'transform_sentence'):
            return np.array([self.transform_sentence(t) for t in texts])
        else:
            raise NotImplementedError()

    @abstractmethod
    def load_pretrained_model(self):
        """ Child classes have to implement this method that will be used
        to load a saved model """
        pass

    def train(self, corpus):
        """ Embedding training method for trainable EmbeddingTransformers """
        raise NotImplementedError('Training is not yet supported for ' +\
                                  self.__class__.__name__)

    def save(self):
        """ Method to save EmbeddingTransformers when trainable """
        raise NotImplementedError('Saving trained embeddings is not yet ' +\
                                  'supported for ' + self.__class__.__name__)


class FastTextTransformer(EmbeddingTransformer):
    """ Facebook FastText embeddings,
    see https://github.com/facebookresearch/fastText for a description """
    def transform_sentence(self, text):
        """ Return the sum of character n-grams representation """
        return self.model.get_sentence_vector(text)

    def load_pretrained_model(self):
        """ fastText model loader """
        return fastText.load_model(self.model_path)

    def download_embeddings(self, url=FASTTEXT_URL):
        """ Download and unzip fasttext pre-trained embeddings """
        zip_file = os.path.join(os.path.dirname(self.model_path),
                                'wiki.simple.zip')
        urllib.request.urlretrieve(url, zip_file)
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(self.model_path))


W2V_EMBEDDINGS_URL = "https://github.com/eyaler/word2vec-slim/raw/master/" +\
                     "GoogleNews-vectors-negative300-SLIM.bin.gz"


class Word2VecTransformer(EmbeddingTransformer):
    """ Word2Vec embeddings class, transforms a corpus to its w2v
    representation matrix

    Notes
    -----
    Google news pre-trained Word2vec download link:
    https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
    Or the lighter version (300k words instead of 3M) here:
    https://github.com/eyaler/word2vec-slim/raw/master/GoogleNews-vectors-negative300-SLIM.bin.gz
    """
    def load_pretrained_model(self):
        """ Load a pre-trained word2vec model """
        if self.model_path.endswith('.bin'):
            w2v = KeyedVectors.load_word2vec_format(self.model_path, binary=True)
        elif self.model_path.endswith('.vec'):
            w2v = KeyedVectors.load_word2vec_format(self.model_path, binary=False)
        else:
            raise NameError('Unknown file extension.')
        return w2v

    def train(self, corpus):
        default_params = dict(size=100, window=2, min_count=5,
                              workers=cpu_count())
        params = {k: self.params.get(k, default) for k, default in
                  default_params.items()}
        x = np.array([text.split() for text in corpus])
        self.model = Word2Vec(x, **params)

    def save(self):
        self.model.save(self.model_path)

    def transform_sentence(self, text):
        """ Compute mean w2v vector for the input text"""
        def preprocess_text(self, text):
            """ Prepare text for Gensim model, excluding unknown words"""
            if not isinstance(text, list):
                if not isinstance(text, str):
                    raise TypeError
                text = text.split()
            return list(filter(lambda x: x in self.model.wv.vocab, text))
        text = preprocess_text(self, text)
        if not text:
            return np.zeros(self.model.vector_size)
        return np.mean(self.model.wv[text], axis=0)

    def download_embeddings(self, url=W2V_EMBEDDINGS_URL, outfile=None):
        """ Download Word2vec pre-computed embeddings from Eyaler github repo """
        gz_file = os.path.join(os.path.dirname(self.model_path),
                               'GoogleNews-vectors-negative300.bin')
        urllib.request.urlretrieve(url, gz_file)
        if outfile is None:
            outfile = gz_file[:-3]
        with gzip.open(gz_file, 'rb') as f_in, open(outfile, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
