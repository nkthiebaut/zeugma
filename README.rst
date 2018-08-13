.. -*- mode: rst -*-

|Python36|_ |TravisBuild|_ |Coveralls|_

.. |Python36| image:: https://img.shields.io/badge/python-3.6-blue.svg
.. _Python36: https://badge.fury.io/py/scikit-learn

.. |TravisBuild| image:: https://travis-ci.org/nkthiebaut/zeugma.svg?branch=master
.. _TravisBuild: https://travis-ci.org/nkthiebaut/zeugma

.. |Coveralls| image:: https://coveralls.io/repos/github/nkthiebaut/zeugma/badge.svg?branch=master
.. _Coveralls: https://coveralls.io/github/nkthiebaut/zeugma?branch=master

======
Zeugma
======

Unified framework for word embeddings (Word2Vec, GloVe, FastText, ...) use in machine learning pipelines, compatible with `scikit-learn Pipelines <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_.

Installation
============

Install package with ``pip install zeugma``.


Examples
========

Embedding transformers can be either be used with downloaded embeddings (they
all come with a default embedding URL) or trained.

Pretrained embeddings
--------------------------------

As an illustrative example the **cosine similarity** of the sentences *zeugma* and *figure of speech* is computed using the GloVeTransformer
with **downloaded embeddings** (default URL is used here)::

    >>> from zeugma.embeddings import EmbeddingTransformer
    >>> glove = EmbeddingTransformer('glove')
    >>> embeddings = glove.transform(['zeugma', 'figure of speech'])
    >>> from sklearn.metrics.pairwise import cosine_similarity
    >>> cosine_similarity(embeddings)[0, 1]
    0.32840478

Training embeddings
-------------------

To train your own Word2Vec embeddings use the `Gensim sklearn API <https://radimrehurek.com/gensim/sklearn_api/w2vmodel.html>`_.


Fine-tuning embeddings
----------------------

Embeddings fine tuning (training embeddings with preloaded values) will be implemented in the future.
