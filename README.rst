.. -*- mode: rst -*-

|Python36|_ |TravisBuild|_

.. |Python36| image:: https://img.shields.io/badge/python-3.6-blue.svg
.. _Python36: https://badge.fury.io/py/scikit-learn

.. |TravisBuild| image:: https://travis-ci.org/nkthiebaut/zeugma.svg?branch=master
.. _TravisBuild: https://travis-ci.org/nkthiebaut/zeugma

======
Zeugma
======

Unified framework for word embeddings (Word2Vec, GloVe, FastText, ...) use in machine learning pipelines, compatible with `scikit-learn Pipelines <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_.

Installation
============

Install package with ``pip install Cython && pip install zeugma`` (Cython is required by the fastText package, on which zeugma is dependent).


Examples
========

Embedding transformers can be either be used with downloaded embeddings (they
all come with a default embedding URL) or trained.

Pretrained downloaded embeddings
--------------------------------

As an illustrative example the **cosine similarity** of the sentences *zeugma* and *figure of speech* is computed using the FastTextTransformer
with **downloaded embeddings** (default URL is used here)::

    >>> from zeugma.embeddings import FastTextTransformer
    >>> model_path = './fasttext.bin'
    >>> FastTextTransformer.download_embeddings(model_path)
    >>> fasttext = FastTextTransformer(model_path)
    >>> embeddings = fasttext.transform(['zeugma', 'figure of speech'])
    >>> from sklearn.metrics.pairwise import cosine_similarity
    >>> cosine_similarity(embeddings)[0, 1]
    0.32840478

Training embeddings
-------------------
Zeugma can also be used to compute the **embeddings on your own corpus** (composed of only two sentences here)::

      >>> from zeugma.embeddings import Word2VecTransformer
      >>> w2v = Word2Word2VecTransformer(trainable=True)
      >>> embeddings = w2v.fit_transform(['zeugma', 'figure of speech'])
      >>> from sklearn.metrics.pairwise import cosine_similarity
      >>> cosine_similarity(embeddings)[0, 1]
      -0.028218582

Fine-tuning embeddings
----------------------

Embeddings fine tuning (training embeddings with preloaded values) will be implemented in the future.
