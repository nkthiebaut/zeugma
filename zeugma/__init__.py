# -*- coding:utf-8 -*-
"""
Created on the 05/01/18
@author: Nicolas Thiebaut
@email: nkthiebaut@gmail.com
"""
from pkg_resources import get_distribution

from .embeddings import FastTextTransformer, GloVeTransformer, Word2VecTransformer
from .texttransformers import RareWordsTagger, ItemSelector, TextStats
from .keras_transformers import TextsToSequences, Padder

__version__ = get_distribution('zeugma').version
