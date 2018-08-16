# -*- coding:utf-8 -*-
"""
Created on the 05/01/18
@author: Nicolas Thiebaut
@email: nkthiebaut@gmail.com
"""

OOV_TAG = '<oov>'  # Out-Of-Vocabulary tag for rare words

DEFAULT_PRETRAINED_EMBEDDINGS = {
    'fasttext': 'fasttext-wiki-news-subwords-300',
    'word2vec': 'word2vec-google-news-300',
    'glove': 'glove-twitter-25',
}
