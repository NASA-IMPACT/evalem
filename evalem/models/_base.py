#!/usr/bin/env python3

from abc import abstractmethod
from typing import Iterable, Type

from transformers import Pipeline as HF_Pipeline
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .._base import AbstractBase
from ..structures import EvaluationPredictionInstance


class ModelWrapper(AbstractBase):
    """
    ModelWrapper is another component of the framework that abstracts away
    all the upstream models into a nice wrapper.

    All the downstream implementation of `ModelWrapper` should implement
    the `predict(...)` method.

    Note:
        In order to convert to task-specific downstream format, we provide
        `_map_predictions(...)` method which user can override. By default,
        it is an identity that doesn't change the format egested by the model.
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
        """
        Entrypoint method for predicting using the wrapped model

        Args:
            ```inputs```
                Represent input dataset whose format depends on
                downstream tasks.

        Returns:
            Iterable of predicted instance
        """
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
    A wrapper for upstream HuggingFace Language Model and corresponding tokenizer.

    Args:
        ```model``` : ```Type[PreTrainedModel]```
            HuggingFace pretrained language model
        ```tokenizer```: ```Type[PreTrainedTokenizerBase]```
            HuggingFace tokenizer
    """

    def __init__(
        self,
        model: Type[PreTrainedModel],
        tokenizer: Type[PreTrainedTokenizerBase],
    ) -> None:
        super().__init__(model=model)
        self.tokenizer = tokenizer


class HFPipelineWrapper(HFWrapper):
    """
    A ModelWrapper to wrap huggingface pipeline which is itself comprised
    of a model and a tokenizer based on some tasks.

    Args:
        ```pipeline```:
            A HuggingFace pipeline object used for prediction

    See `evalem.models.defaults.DefaultQAModelWrapper` for a downstream
    implementation.

    Direct usage:

        .. code-block: python

            from transformers import pipeline as hf_pipeline
            from evalem.models import HFPipelineWrapper

            pipe = hf_pipeline("question-answering")
            wrapped_model = HFPipelineWrapper(pipe)

            # compute predictions
            # (format?) and pass to evaluator along with references
            predictions = wrapped_model.predict(<inputs>)
    """

    def __init__(self, pipeline: Type[HF_Pipeline], debug: bool = False) -> None:
        """
        Args:
            ```pipeline```:
                A HuggingFace pipeline object used for prediction
        """
        super().__init__(pipeline)

    def predict(self, inputs, **kwargs):
        return self._map_predictions(self.model(inputs))


def main():
    pass


if __name__ == "__main__":
    main()
