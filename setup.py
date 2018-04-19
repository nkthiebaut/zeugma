#!/usr/bin/env python

from setuptools import setup
import sys

try:
    import Cython
except ImportError:
    print('Cython is required during installation')
    sys.exit(1)

long_description = """
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

As an illustrative example the **cosine similarity** of the sentences *zeugma* and *figure of speech* is computed using the GloVeTransformer
with **downloaded embeddings** (default URL is used here)::

    >>> from zeugma.embeddings import GloVeEmbeddings
    >>> GloVeTransformer.download_embeddings()
    >>> glove = GloVeTransformer(model_path)
    >>> embeddings = GloVe.transform(['zeugma', 'figure of speech'])
    >>> from sklearn.metrics.pairwise import cosine_similarity
    >>> cosine_similarity(embeddings)[0, 1]
    0.32840478

Training embeddings
-------------------
Zeugma can also be used to compute the **embeddings on your own corpus** (composed of only two sentences here)::

      >>> from zeugma.embeddings import Word2VecTransformer
      >>> w2v = Word2VecTransformer(trainable=True)
      >>> embeddings = w2v.fit_transform(['zeugma', 'figure of speech'])
      >>> from sklearn.metrics.pairwise import cosine_similarity
      >>> cosine_similarity(embeddings)[0, 1]
      -0.028218582

Fine-tuning embeddings
----------------------

Embeddings fine tuning (training embeddings with preloaded values) will be implemented in the future.
"""

setup(name='zeugma',
      packages=['zeugma'],
      version='0.41',
      license='MIT',
      description="Unified framework for word embeddings (Word2Vec, GloVe, FastText, ...) compatible with scikit-learn Pipeline",
      long_description=long_description,
      long_description_content_type='text/x-rst',
      author='Nicolas Thiebaut',
      author_email='nkthiebaut@gmail.com',
      url='https://github.com/nkthiebaut',
      download_url='https://github.com/nkthiebaut/zeugma/archive/0.41.tar.gz',
      keywords=['embeddings'],
      classifiers=[],
      setup_requires=[
          'pytest-runner',
          'numpy>=1.13.3',
          'Cython>=0.27.3',
      ],
      install_requires=[
          'numpy>=1.13.3',
          'Cython>=0.27.3',
          'pandas>=0.20.3',
          'gensim>=3.2.0',
          'scikit_learn>=0.19.1',
          'tensorflow>=1.5.0',
          'keras>=2.1.3',
          'fastText',
      ],
      tests_require=['pytest>=3.3.2'],
      dependency_links=['git+https://github.com/facebookresearch/fastText.git'],
      )
