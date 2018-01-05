#!/usr/bin/env python

from distutils.core import setup

setup(name='zeugma',
      version='0.1',
      description='Unified framework for word embeddings (Word2Vec, GloVe, FastText, ...) compatible with scikit-learn Pipeline',
      author='Nicolas Thiebaut',
      author_email='nkthiebaut@gmail.com',
      url='https://github.com/nkthiebaut',
      download_url='https://github.com/nkthiebaut/zeugma/archive/0.1.tar.gz',
      install_requires=[
          'numpy>=0.1',
	  'pandas',
          'sklearn',
      ]
     )

