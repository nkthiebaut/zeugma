.. -*- mode: rst -*-

|Python36| |TravisBuild| |Coveralls| |ReadTheDocs|

.. |Python36| image:: https://img.shields.io/badge/python-3.6-blue.svg
    :target: https://badge.fury.io/py/scikit-learn

.. |TravisBuild| image:: https://travis-ci.org/nkthiebaut/zeugma.svg?branch=master&service=github
    :target: https://travis-ci.org/nkthiebaut/zeugma

.. |Coveralls| image:: https://img.shields.io/coveralls/github/nkthiebaut/zeugma.svg
    :target: https://coveralls.io/github/nkthiebaut/zeugma?branch=master

.. |ReadTheDocs| image:: https://readthedocs.org/projects/zeugma/badge/ 
    :target: https://readthedocs.org/projects/zeugma/

======
Zeugma
======

.. inclusion-marker-do-not-remove

📝 Natural language processing (NLP) utils: word embeddings (Word2Vec, GloVe, FastText, ...) and preprocessing transformers, compatible with `scikit-learn Pipelines <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_. 🛠


Installation
------------

Install package with ``pip install zeugma``.


Examples
--------

Embedding transformers can be either be used with downloaded embeddings (they
all come with a default embedding URL) or trained.

Pretrained embeddings
*********************

As an illustrative example the cosine similarity of the sentences *what is zeugma* and *a figure of speech* is computed using the `GloVe <https://nlp.stanford.edu/projects/glove/>`_ pretrained embeddings.::

    >>> from zeugma.embeddings import EmbeddingTransformer
    >>> glove = EmbeddingTransformer('glove')
    >>> embeddings = glove.transform(['what is zeugma', 'a figure of speech'])
    >>> from sklearn.metrics.pairwise import cosine_similarity
    >>> cosine_similarity(embeddings)[0, 1]
    0.8721696

Training embeddings
*******************

To train your own Word2Vec embeddings use the `Gensim sklearn API <https://radimrehurek.com/gensim/sklearn_api/w2vmodel.html>`_.


Fine-tuning embeddings
**********************

Embeddings fine tuning (training embeddings with preloaded values) will be implemented in the future.
