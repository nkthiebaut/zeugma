import os
from typing import Iterable, List, Union

import gensim.downloader as api
import numpy as np
from gensim.models.keyedvectors import Word2VecKeyedVectors
from sklearn.base import BaseEstimator, TransformerMixin
from zeugma.conf import DEFAULT_PRETRAINED_EMBEDDINGS
from zeugma.logger import package_logger as logger


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Text vectorizer class: load pre-trained embeddings and transform texts
    into vectors.
    """

    model: Word2VecKeyedVectors
    aggregation: str

    def __init__(self, model: str = "glove", aggregation: str = "average"):
        """Load pre-trained embeddings, either locally if model is a local file path
        or a Word2VecKeyedVector object, or downloaded from the gensim API if a string
        is provided.
        """
        if aggregation not in {"average", "sum", "minmax"}:
            raise ValueError(
                f"Unknown embeddings aggregation mode: {aggregation}, the available "
                "ones are: average, sum, or minmax."
            )
        if isinstance(model, str):
            model = model.lower()
            if model in DEFAULT_PRETRAINED_EMBEDDINGS.keys():
                model_gensim_name = DEFAULT_PRETRAINED_EMBEDDINGS[model]
                self.model = api.load(model_gensim_name)
            elif model in api.info()["models"].keys():
                self.model = api.load(model)  # pragma: no cover
            elif os.path.exists(model):
                logger.info("Loading local model")
                self.model = Word2VecKeyedVectors.load(model)
                if not isinstance(self.model, Word2VecKeyedVectors):
                    raise TypeError(
                        "The input model should be a Word2VecKeyedVectors object but "
                        f"it is a {type(self.model)} object."
                    )
            else:
                raise KeyError(
                    f"Unknown pre-trained model name: {model}. Available models are"
                    + ", ".join(api.info()["models"].keys())
                )
            logger.info("Loaded model keyed vectors: " + model)
        elif isinstance(model, Word2VecKeyedVectors):
            self.model = model
            logger.info("Loaded model keyed vectors.")
        else:
            raise TypeError(
                "Input pre-trained model should be a string or a gensim "
                "Word2VecKeyedVectors object"
            )
        self.aggregation = aggregation
        self.embedding_dimension = self.model.vector_size
        if self.aggregation == "minmax":
            self.embedding_dimension *= 2

    def transform_sentence(self, text: Union[Iterable, str]) -> np.array:
        """Compute an aggregate embedding vector for an input str or iterable of str."""

        def preprocess_text(raw_text: Union[Iterable, str]) -> List[str]:
            """Prepare text for the model, excluding unknown words"""
            if not isinstance(raw_text, list):
                if not isinstance(raw_text, str):
                    raise TypeError(
                        f"Input should be a str or a list of str, got {type(raw_text)}"
                    )
                raw_tokens = raw_text.split()
            return list(filter(lambda x: x in self.model, raw_tokens))

        tokens = preprocess_text(text)

        if not tokens:
            return np.zeros(self.embedding_dimension, dtype=np.float32)

        if self.aggregation == "average":
            return np.mean(self.model[tokens], axis=0)
        elif self.aggregation == "sum":
            return np.sum(self.model[tokens], axis=0)
        elif self.aggregation == "minmax":
            maxi = np.max(self.model[tokens], axis=0)
            mini = np.min(self.model[tokens], axis=0)
            return np.append(mini, maxi)

    def fit(self, x: Iterable[Iterable], y: Iterable = None) -> BaseEstimator:
        """Has to define fit method to conform scikit-learn Transformer
        definition and integrate a sklearn.Pipeline object"""
        return self  # pragma: no cover

    def transform(self, texts: Iterable[str]) -> Iterable[Iterable]:
        """Transform corpus from single text transformation method"""
        # TODO: parallelize this method with multiprocessing
        return np.array([self.transform_sentence(t) for t in texts])
