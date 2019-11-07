# -*- coding:utf-8 -*-
"""
Created on the 05/01/18
@author: Nicolas Thiebaut
@email: nkthiebaut@gmail.com
"""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def sample_corpus():
    """ Return a sample corpus in a numpy.array """
    corpus = [
        "Here a first example text",
        "This is a second text with a weird word gwiurgergwggreg",
        "This is a second text with a weird word",
        "Et c'est un troisieme text avec un accent",
    ]
    return np.array(corpus)


@pytest.fixture(scope="module")
def sample_corpus_embedding():
    """ Return a sample corpus in a numpy.array """
    corpus = [
        "human computer",
        "interface gwiurgergwggreg",
        "interface",
        "Et c'est un troisième text avec un accent",
    ]
    return np.array(corpus)
