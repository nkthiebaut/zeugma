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

Install package with ``pip install Cython && pip install zeugma`` (Cython is required by the fastText package, on which zeugma is dependent).

-------
Example
-------

Embedding transformers can be either be used with downloaded embeddings (they
all come with a default embedding URL) or trained.

Example with a pretrained downloaded model (default URL is used here)::

    >>> from zeugma.embeddings import FastTextTransformer
    >>> model_path = './fasttext.bin'
    >>> FastTextTransformer.download_embeddings(model_path)
    >>> fasttext = FastTextTransformer(model_path)
    >>> v1 = fasttext.transform(['zeugma'])
    >>> v2 = fasttext.transform(['figure of speech'])
    >>> from sklearn.metrics.pairwise import cosine_similarity
    >>> cosine_similarity(v1, v2)[0][0]
    0.32840478

Example training model::

      >>> from zeugma.embeddings import Word2VecTransformer
      >>> w2v = Word2Word2VecTransformer(trainable=True)
      >>> w2v.fit(['zeugma', 'figure of speech'])
      >>> v1 = w2v.transform(['zeugma'])
      >>> v2 = w2v.transform(['figure of speech'])
      >>> from sklearn.metrics.pairwise import cosine_similarity
      >>> cosine_similarity(v1, v2)[0][0]
      -0.028218582


Embeddings fine tuning (training embeddings with preloaded values) will be implemented in the future.
