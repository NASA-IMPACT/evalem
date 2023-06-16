#!/usr/bin/env python3

from abc import abstractmethod
from typing import Callable, Iterable

# TODO implement wrapper for these
# from transformers import Pipeline as HF_Pipeline
# from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .abc import AbstractBase
from .structures import EvaluationPredictionInstance


class ModelWrapper(AbstractBase):
    """
    ModelWrapper is another component of the framework that abstracts away
    all the upstream models into a nice wrapper.

    All the downstream implementation of `ModelWrapper` should implement
    the `_predict(...)` method which is itself called by `.predict(...)` method.

    Args:
        ```model```:
            Input model that's being wrapped for common interface
        ```debug```: ```bool```
            If enabled, debugging logs could be printed
        ```kwargs```:
            - ```inputs_preprocessor```
                A `Callable` to apply on inputs.
            - ```predictions_postprocessor```
                A `Callable` to apply on model outputs/predictions.

    Note:
        - Override `_preprocess_inputs` method to change data format for
            model input. Default it identity (no change).
        - Override `_postprocess_predictions` to convert predictions to
            task-specific downstream format. Defaults to identity (no change).
    """

    def __init__(
        self,
        model,
        debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(debug=debug)
        self.model = model

        # specifies how the input format conversion is done
        self.inputs_preprocessor: Callable = (
            kwargs.get("inputs_preprocessor", self._preprocess_inputs)
            or self._preprocess_inputs
        )

        # specifies how the predictions formatting is done
        self.predictions_postprocessor: Callable = (
            kwargs.get("predictions_postprocessor", self._postprocess_predictions)
            or self._postprocess_predictions
        )

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
        inputs = self.inputs_preprocessor(inputs, **kwargs)
        predictions = self._predict(inputs, **kwargs)
        return self.predictions_postprocessor(predictions, **kwargs)

    @abstractmethod
    def _predict(
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

    def _preprocess_inputs(self, inputs: Iterable, **kwargs) -> Iterable:
        """
        A helper method to transform inputs suitable for model to ingest.
        By default, it's an identity function.
        """
        return inputs

    def _postprocess_predictions(self, predictions: Iterable, **kwargs):
        """
        A helper method to transform predictions from the models
        into any downstream format. By default, it's an identity function.
        """
        # default -> Identity
        return predictions

    def __call__(
        self,
        inputs: Iterable,
        **kwargs,
    ) -> Iterable[EvaluationPredictionInstance]:
        return self.predict(inputs, **kwargs)


class HFWrapper(ModelWrapper):
    """
    A type wrapper for all the downstream Huggingface based models
    """

    pass


def main():
    pass


if __name__ == "__main__":
    main()
