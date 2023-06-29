#!/usr/bin/env python3

from typing import Iterable, List, Optional, Union

from transformers import pipeline as hf_pipeline

from ..._base.structures import ClassificationDTO

# load nlp specific structure dto
from ..structures import QuestionAnsweringDTO
from ._base import (
    HFORTMixin,
    HFPipelineWrapper,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


class QuestionAnsweringHFPipelineWrapper(HFPipelineWrapper, HFORTMixin):
    """
    A HFPipelineWrapper for question-answering.

    Args:
        ```model```: ```Type[PreTrainedModel]```
            Which model to use?
        ```tokenizer```: ```Type[PreTrainedTokenizerBase]```
            Which tokenizer to use?
        ```device```:```str```
            Which device to run the model on? cpu? gpu? mps?

    Usage:
        .. code-block: python

                from evalem.nlp.models import QuestionAnsweringHFPipelineWrapper

                model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")

                wrapped_model = QuestionAnsweringHFPipelineWrapper(
                    model=model,
                    tokenizer=tokenizer,
                    device="mps"
                )

                data = [dict(context="This is a context.", question="What is this?")]

                res = wrapped_model(data)
                print(res[0])

    """

    _task = "question-answering"

    def __init__(
        self,
        model: Optional[
            Union[str, PreTrainedModel]
        ] = "distilbert-base-cased-distilled-squad",
        tokenizer: Optional[Union[str, PreTrainedTokenizerBase]] = None,
        device: str = "cpu",
        hf_params: Optional[dict] = None,
        **kwargs,
    ) -> None:
        self.hf_params = hf_params or {}
        super().__init__(
            pipeline=hf_pipeline(
                self._task,
                model=model,
                tokenizer=tokenizer,
                device=device,
                **self.hf_params,
            ),
            **kwargs,
        )

    def _postprocess_predictions(
        self,
        predictions: Union[dict, List[dict]],
        **kwargs,
    ) -> Iterable[QuestionAnsweringDTO]:
        """
        This method converts the pipeline's default output format
        to the iterable of QuestionAnsweringDTO.

        Args:
            ```predictions```: ```Union[dict, List[dict]]```
                Predictions provided by the QA pipeline.

        Returns:
            Converted format: ```Iterable[QuestionAnsweringDTO]```
        """
        if isinstance(predictions, dict):
            predictions = [predictions]

        # Note: Default model here is guaranteed to have these keys.
        return list(
            map(
                lambda p: QuestionAnsweringDTO(
                    value=p.get("value", p["answer"]),
                    score=p["score"],
                    start=p.get("start"),
                    end=p.get("end"),
                    context=p.get("context"),
                    question=p.get("question"),
                ),
                predictions,
            ),
        )


class DefaultQAModelWrapper(HFPipelineWrapper):
    """
    Deprecated: Use `QuestionAnsweringHFPipelineWrapper()`
    """

    def __init__(self, device: str = "cpu") -> None:
        raise DeprecationWarning(
            "Deprecated ModelWrapper. Please use `QuestionAnsweringHFPipelineWrapper`",
        )


class TextClassificationHFPipelineWrapper(HFPipelineWrapper, HFORTMixin):
    """
    A HFPipelineWrapper for text classification.

    Args:
        ```model```: ```Type[PreTrainedModel]```
            Which model to use?
        ```tokenizer```: ```Type[PreTrainedTokenizerBase]```
            Which tokenizer to use?
        ```device```:```str```
            Which device to run the model on? cpu? gpu? mps?

    Usage:
        .. code-block: python

                from evalem.nlp.models import QuestionAnsweringHFPipelineWrapper

                model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

                wrapped_model = TextClassificationHFPipelineWrapper(
                    model=model,
                    tokenizer=tokenizer,
                    device="mps"
                )

                data = ["evalem is awesome."]

                res = wrapped_model(data)
                print(res[0])
    """

    _task = "text-classification"

    def __init__(
        self,
        model: Optional[
            Union[str, PreTrainedModel]
        ] = "distilbert-base-uncased-finetuned-sst-2-english",
        tokenizer: Optional[Union[str, PreTrainedTokenizerBase]] = None,
        device: str = "cpu",
        hf_params: Optional[dict] = None,
        **kwargs,
    ) -> None:
        self.hf_params = hf_params or {}
        super().__init__(
            pipeline=hf_pipeline(
                self._task,
                model=model,
                tokenizer=tokenizer,
                device=device,
                **self.hf_params,
            ),
            **kwargs,
        )
        # mapping  from int code to actual label name.
        self.label_map = kwargs.get("label_map", {})

    def _postprocess_predictions(
        self,
        predictions: Union[dict, List[dict]],
    ) -> Iterable[ClassificationDTO]:
        """
        This method converts the pipeline's default output format
        to the iterable of ClassificationDTO.

        Args:
            ```predictions```: ```Union[dict, List[dict]]```
                Predictions provided by the the classificaton pipeline.

        Returns:
            Converted format: ```Iterable[ClassificationDTO]```
        """
        if isinstance(predictions, dict):
            predictions = [predictions]

        # Note: Default model here is guaranteed to have these keys.
        # Use label mapping. If mapping doesn't exist, just use the prediction.
        predictions = map(
            lambda p: ClassificationDTO(
                value=self.label_map.get(p["label"], p["label"]),
                score=p.get("score"),
            ),
            predictions,
        )
        return list(predictions)


def main():
    pass


if __name__ == "__main__":
    main()
