#!/usr/bin/env python

from setuptools import setup

setup(
    name="zeugma",
    packages=["zeugma"],
    version="0.45",
    python_requires=">=3.6",
    license="MIT",
    description="""Natural language processing (NLP) utils: word embeddings (Word2Vec,
    GloVe, FastText, ...) and preprocessing transformers, compatible with scikit-learn
    Pipelines.""",
    long_description=open("README.rst", "r").read(),
    long_description_content_type="text/x-rst",
    author="Nicolas Thiebaut",
    author_email="nkthiebaut@gmail.com",
    url="https://github.com/nkthiebaut",
    download_url="https://github.com/nkthiebaut/zeugma/archive/0.45.tar.gz",
    classifiers=[],
    setup_requires=["pytest-runner", "numpy>=1.13.3", "Cython>=0.27.3"],
    install_requires=[
        "numpy>=1.13.3",
        "Cython>=0.27.3",
        "pandas>=0.20.3",
        "gensim>=3.5.0",
        "scikit_learn>=0.19.1",
        "tensorflow>=1.5.0",
        "keras>=2.1.3",
    ],
    tests_require=["pytest>=3.3.2"],
)
