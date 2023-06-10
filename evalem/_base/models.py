#!/usr/bin/env python3

from abc import abstractmethod
from typing import Callable, Iterable, Type

from transformers import Pipeline as HF_Pipeline
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ._base import AbstractBase
from ..nlp.structures import EvaluationPredictionInstance


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
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
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

            # Or: if you want to specify how to post-process predictions,
            # provide the processor explicitly.
            wrapped_model = HFPipelineWrapper(
                pipeline("question-answering", model="deepset/roberta-base-squad2"),
                predictions_postprocessor=lambda xs: list(map(lambda x: x["answer"], xs))
            )


            # compute predictions
            # (format?) and pass to evaluator along with references
            predictions = wrapped_model.predict(<inputs>)
    """

    def __init__(self, pipeline: Type[HF_Pipeline], **kwargs) -> None:
        """
        Args:
            ```pipeline```:
                A HuggingFace pipeline object used for prediction
        """
        super().__init__(model=pipeline, **kwargs)

    def _predict(self, inputs, **kwargs):
        return self.model(inputs, **kwargs)

    @property
    def pipeline(self) -> HF_Pipeline:
        return self.model


def main():
    pass


if __name__ == "__main__":
    main()
