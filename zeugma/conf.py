# -*- coding:utf-8 -*-
"""
Created on the 05/01/18
@author: Nicolas Thiebaut
@email: nkthiebaut@gmail.com
"""

OOV_TAG = '<oov>'  # Out-Of-Vocabulary tag for rare words

DEFAULT_PRETRAINED_EMBEDDINGS = {
    'FastText': 'fasttext-wiki-news-subwords-300',
    'Word2Vec': 'word2vec-google-news-300',
    'GloVe': 'glove-twitter-25',
}
