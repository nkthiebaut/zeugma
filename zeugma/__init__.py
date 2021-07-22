# -*- coding:utf-8 -*-
"""
Created on the 05/01/18
@author: Nicolas Thiebaut
@email: nkthiebaut@gmail.com
"""
# from pkg_resources import get_distribution

from .embeddings import EmbeddingTransformer  # noqa
from .texttransformers import RareWordsTagger, ItemSelector, TextStats  # noqa
from .keras_transformers import TextsToSequences, Padder  # noqa

__version__ = "0.49"
