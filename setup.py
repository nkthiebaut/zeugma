#!/usr/bin/env python

from distutils.core import setup

setup(name='zeugma',
      packages=['zeugma'],
      version='0.16',
      license='MIT',
      description='Unified framework for word embeddings (Word2Vec, GloVe, FastText, ...) compatible with scikit-learn Pipeline',
      author='Nicolas Thiebaut',
      author_email='nkthiebaut@gmail.com',
      url='https://github.com/nkthiebaut',
      download_url='https://github.com/nkthiebaut/zeugma/archive/0.16.tar.gz',
      keywords=['embeddings'],
      classifiers=[],
      install_requires=[
          'pandas>=0.20.3',
          'numpy>=1.13.3',
          'gensim>=3.2.0',
          'scikit_learn>=0.19.1',
          'Cython',
          'fastText',
      ],
      dependency_links=['git+git://github.com/facebookresearch/fastText.git']
     )

