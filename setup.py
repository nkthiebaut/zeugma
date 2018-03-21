#!/usr/bin/env python

from distutils.core import setup
import sys

try:
    import Cython
except ImportError:
    print('Cython is required during installation')
    sys.exit(1)

setup(name='zeugma',
      packages=['zeugma'],
      version='0.38',
      license='MIT',
      description="Unified framework for word embeddings (Word2Vec, GloVe, FastText, ...) compatible with scikit-learn Pipeline",
      author='Nicolas Thiebaut',
      author_email='nkthiebaut@gmail.com',
      url='https://github.com/nkthiebaut',
      download_url='https://github.com/nkthiebaut/zeugma/archive/0.38.tar.gz',
      keywords=['embeddings'],
      classifiers=[],
      setup_requires=[
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
