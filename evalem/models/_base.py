#!/usr/bin/env python3

from abc import abstractmethod
from typing import Iterable

from transformers import pipeline as hf_pipeline

from .._base import AbstractBase
from ..structures import EvaluationPredictionInstance


class ModelWrapper(AbstractBase):
    """
    ModelWrapper is another component of the framework that abstracts away
    all the upstream models into a nice wrapper.

    All the downstream implementation of `ModelWrapper` should implement
    the `predict(...)` method.
    """

    def __init__(self, model, debug: bool = False, **kwargs) -> None:
        super().__init__(debug=debug)
        self.model = model

    @abstractmethod
    def predict(
        self,
        inputs: Iterable,
        **kwargs,
    ) -> Iterable[EvaluationPredictionInstance]:
        raise NotImplementedError()

    def __call__(
        self,
        inputs: Iterable,
        **kwargs,
    ) -> Iterable[EvaluationPredictionInstance]:
        return self.predict(inputs, **kwargs)

    def _map_predictions(self, predictions: Iterable):
        """
        A helper method to transform predictions from the models
        into any downstream format. By default, it's an identity function.
        """
        # default -> Identity
        return predictions


class HFWrapper(ModelWrapper):
    """
    A type wrapper for all the downstream Huggingface based models
    """

    pass


class HFLMWrapper(HFWrapper):
    """
    A wrapper for upstream HuggingFace model and corresponding tokenizer.
    """

    def __init__(self, model, tokenizer) -> None:
        super().__init__(model=model)
        self.tokenizer = tokenizer


class HFPipelineWrapper(HFWrapper):
    """
    A ModelWrapper to wrap huggingface pipeline which is itself comprised
    of a model and a tokenizer based on some tasks.

    Args:
        ```pipeline```:
            A HuggingFace pipeline object
    """

    def __init__(self, pipeline, debug: bool = False) -> None:
        super().__init__(pipeline)

    def predict(self, inputs, **kwargs):
        return self._map_predictions(self.model(inputs))


def main():
    pass


if __name__ == "__main__":
    main()
