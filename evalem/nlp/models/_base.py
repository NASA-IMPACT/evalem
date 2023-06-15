#!/usr/bin/env python3
"""
    This module contains nlp specific base classes.
"""

from typing import Type

from ..._base.models import (
    HF_Pipeline,
    HFWrapper,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


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
