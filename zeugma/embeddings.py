# -*- coding:utf-8 -*-
"""
Created on the 05/01/18
@author: Nicolas Thiebaut
@email: nkthiebaut@gmail.com
"""

from abc import ABCMeta, abstractmethod
from functools import reduce
import gzip
from multiprocessing import cpu_count
import os
import shutil
import urllib
import zipfile

from gensim.models import KeyedVectors, Word2Vec
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from zeugma.logger import PACKAGE_LOGGER as logger
from zeugma.conf import MODELS_DIR, DEFAULT_PRETRAINED_EMBEDDINGS


class EmbeddingTransformer(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    """ Abstract text vectorizer class """

    def __init__(self, model_path=None, trainable=False, **kwargs):
        self.trainable = trainable
        self.model_path = model_path
        self.params = dict(kwargs)
        if trainable:
            for method in ['train', 'save']:
                if not hasattr(self, method):
                    raise NotImplementedError('trainable is set to True but ' +\
                            self.__class__.__name__ + ' does not implement ' + \
                                         method + ' method.')
        else:
            if model_path is None:
                logger.info('Embedding is not trainable and model_path not' +\
                            'specified, using default model location:' + MODELS_DIR)
                self.model_path = os.path.join(MODELS_DIR, self.__class__.default_model_path)
            if not hasattr(self, 'load_pretrained_model'):
                raise NotImplementedError(self.__class__.__name__ + \
                                          ' does not support pretrained models.')
            else:
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(self.__class__.__name__ +
                                            ' model file not found')
            self.model = self.load_pretrained_model()

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
            if self.model_path is not None:
                self.save()
        return self

    def transform(self, texts):
        """ Transform corpus from single text transformation method """
        if hasattr(self, 'transform_sentence'):
            return np.array([self.transform_sentence(t) for t in texts])
            # TODO: parallelize this method with multiprocessing
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

    default_model_path = os.path.join(MODELS_DIR,
                                      DEFAULT_PRETRAINED_EMBEDDINGS['Word2Vec']['filename'])

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
    
    @staticmethod
    def download_embeddings(model_path=MODELS_DIR + os.sep,
                            url=DEFAULT_PRETRAINED_EMBEDDINGS['Word2Vec']['url'],
                            outfile=None):
        """ Download Word2vec pre-computed embeddings from Eyaler github repo """
        gz_file = os.path.join(os.path.dirname(model_path),
                               'GoogleNews-vectors-negative300.bin')
        urllib.request.urlretrieve(url, gz_file)
        if outfile is None:
            outfile = gz_file[:-3]
        with gzip.open(gz_file, 'rb') as f_in, open(outfile, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


class GloVeTransformer(EmbeddingTransformer):
    """ Glove embeddings class, transforms a corpus to its Glove
    representation matrix"""

    default_model_path = os.path.join(MODELS_DIR,
                                      DEFAULT_PRETRAINED_EMBEDDINGS['GloVe']['filename'])

    def load_pretrained_model(self):
        """ Load a pre-trained GloVe model """
        if self.model_path.endswith('.txt'):
            embeddings_dict = dict()
            with open(self.model_path, encoding="utf8") as glove_stream:
                for line in glove_stream:
                    values = line.split()
                    word = values[0]
                    value = np.asarray(values[1:], dtype='float32')
                    embeddings_dict[word] = value
        else:
            raise NameError('Unknown file extension.')
        return embeddings_dict

    def transform_sentence(self, text):
        """ Return the mean of the words embeddings """
        size = len(self.model['the'])
        embeddings = (self.model.get(w, np.zeros(size)) for w in text)
        text_vector = reduce(np.add, embeddings, np.zeros(size))
        return text_vector / len(text)

    @staticmethod
    def download_embeddings(model_path=MODELS_DIR + os.sep,
                            url=DEFAULT_PRETRAINED_EMBEDDINGS['GloVe']['url']):
        """ Download GloVe pre-computed embeddings from Stanford website """
        model_dir = os.path.dirname(model_path)
        zip_file = os.path.join(model_dir, 'glove.6B.zip')
        urllib.request.urlretrieve(url, zip_file)
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(model_dir)

    #TODO: add GloVe training with the glove_python library
