.. -*- mode: rst -*-

|Python36| |TravisBuild| |Coveralls| |ReadTheDocs| |LGTM| |Black|

.. |Python36| image:: https://img.shields.io/badge/python-3.6-blue.svg
    :target: https://badge.fury.io/py/scikit-learn

.. |TravisBuild| image:: https://travis-ci.org/nkthiebaut/zeugma.svg?branch=master&service=github
    :target: https://travis-ci.org/nkthiebaut/zeugma

.. |Coveralls| image:: https://img.shields.io/coveralls/github/nkthiebaut/zeugma.svg
    :target: https://coveralls.io/github/nkthiebaut/zeugma?branch=master

.. |ReadTheDocs| image:: https://readthedocs.org/projects/zeugma/badge/ 
    :target: https://readthedocs.org/projects/zeugma/

.. |LGTM| image:: https://img.shields.io/lgtm/grade/python/g/nkthiebaut/zeugma.svg?logo=lgtm
    :target: https://lgtm.com/projects/g/nkthiebaut/zeugma/context:python

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black

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


Other examples
**************

Additional examples using Zeugma can be found `in some posts of my blog <https://data4thought.com>`_.


Contribute
----------

Feel free to fork this repo and submit a Pull Request. 

Development
***********

The development workflow for this repo is the following:
1. create a virtual environment:
```python 
python -m venv venv
source venv/bin/activate
```
2. install required packages: `pip install -r requirements.txt`
3. install the pre-commit hooks: `pre-commit install`
4. run the test suite with: `pytest` from the root folder

Distribution via PyPI
*********************

To upload a new version to PyPI, simply:
1. tag your new version on git: `git tag -a x.x -m "my tag message"`
2. update the download_url field in the `setup.py` file
3. commit, push the code and the tag (`git push origin x.x`), and make a PR
4. once the updated code is present in master run `python3 setup.py sdist bdist_wheel` from the root of the package to distribute it.
