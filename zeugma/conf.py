# -*- coding:utf-8 -*-
"""
Created on the 05/01/18
@author: Nicolas Thiebaut
@email: nkthiebaut@gmail.com
"""
import os
from zeugma.logger import PACKAGE_LOGGER as logger

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'zeugma_embeddings')
if not os.path.exists(MODELS_DIR):
    logger.info(MODELS_DIR + ' not found, creating it.')
    os.mkdir(MODELS_DIR)

OOV_TAG = '<oov>'  # Out-Of-Vocabulary tag for rare words


DEFAULT_PRETRAINED_EMBEDDINGS = {'Word2Vec': ()}