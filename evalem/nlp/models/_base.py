#!/usr/bin/env python3
"""
    This module contains nlp specific base classes.
"""


from pathlib import Path
from typing import Optional, Type, Union

from loguru import logger
from transformers import Pipeline as HF_Pipeline  # noqa
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..._base.models import HFWrapper


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


class HFORTMixin:
    from optimum.onnxruntime import (
        ORTModelForQuestionAnswering,
        ORTModelForSequenceClassification,
    )

    _mapping = {
        "question-answering": ORTModelForQuestionAnswering,
        "text-classification": ORTModelForSequenceClassification,
    }

    @classmethod
    def from_onnx(
        cls,
        model,
        tokenizer: Optional[Union[str, PreTrainedTokenizerBase]] = None,
        device: str = "cpu",
        hf_params: Optional[dict] = None,
        **kwargs,
    ) -> Type[HFPipelineWrapper]:
        """
        classmethod to load model based on onnxruntime models.
        It nicely fits with existing huggingface pipeline.
        So, this method maps the standard HF `PreTrainedModel` to
        corresponding `ORTModel`

        Args:
            ```model```: ```Union[str, Type[ORTModel]]```
                Which model to use?
            ```tokenizer```: ```Union[str, Type[PreTrainedTokenizerBase]]```
                Which tokenizer to use?
            ```device```:```str```
                Which device to run the model on? cpu? gpu? mps?

        Returns:
            `Type[HFPipelineWrapper]` object

        Example Usage:
            see `evalem.nlp.models.QuestionAnsweringHFPipelineWrapper`

            .. code-block: python

                    from evalem.nlp.models import QuestionAnsweringHFPipelineWrapper

                    wrapped_model = QuestionAnsweringHFPipelineWrapper.from_onnx(
                        model="tmp/onnx/model_x/",
                        tokenizer="tmp/onnx/model_x/",
                        device="cpu"
                    )

        Note:
            Alternatively, we can directly make use of original constructor by passing
            in the ORTModel with `model=<ORTModel object>` instance as well.

            .. code-block: python

                    from optimum.onnxruntime import ORTModelForQuestionAnswering
                    from evalem.nlp.models import QuestionAnsweringHFPipelineWrapper

                    # here we can directly using original constructor
                    wrapped_model = QuestionAnsweringHFPipelineWrapper(
                        model=ORTModelForQuestionAnswering.from_pretrained("tmp/onnx/model_x"),
                        tokenizer="tmp/onnx/model_x/",
                        device="cpu"
                    )
        """
        task = "question-answering"
        if hasattr(cls, "_task"):
            task = cls._task

        model_cls = cls._mapping[task]

        logger.warning(f"ONNX runtime-based model[{model_cls}]")

        model = (
            model_cls.from_pretrained(model)
            if isinstance(model, (str, Path))
            else model
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            hf_params=hf_params,
            **kwargs,
        )
