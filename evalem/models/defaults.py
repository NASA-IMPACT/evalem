#!/usr/bin/env python3

from typing import Iterable, List, Optional, Union

from transformers import pipeline as hf_pipeline

from ..structures import QAPredictionDTO
from ._base import HFPipelineWrapper, PreTrainedModel, PreTrainedTokenizerBase


class HFPipelineWrapperForQuestionAnswering(HFPipelineWrapper):
    """
    A HFPipelineWrapper for question-answering.

    Args:
        ```model```: ```Type[PreTrainedModel]```
            Which model to use?
        ```tokenizer```: ```Type[PreTrainedTokenizerBase]```
            Which tokenizer to use?
        ```device```:```str```
            Which device to run the model on? cpu? gpu? mps?
    """

    def __init__(
        self,
        model: Optional[
            Union[str, PreTrainedModel]
        ] = "distilbert-base-cased-distilled-squad",
        tokenizer: Optional[
            Union[str, PreTrainedTokenizerBase]
        ] = "distilbert-base-cased-distilled-squad",
        device: str = "cpu",
    ) -> None:
        super().__init__(
            pipeline=hf_pipeline(
                "question-answering",
                model=model,
                tokenizer=tokenizer,
                device=device,
            ),
        )

    def _postprocess_predictions(
        self,
        predictions: Union[dict, List[dict]],
    ) -> Iterable[QAPredictionDTO]:
        """
        This method converts the pipeline's default output format
        to the iterable of QAPredictionDTO.

        Args:
            ```predictions```: ```Union[dict, List[dict]]```
                Predictions provided by the QA pipeline.

        Returns:
            Converted format: ```Iterable[QAPredictionDTO]```
        """
        if isinstance(predictions, dict):
            predictions = [predictions]

        # Note: Default model here is guaranteed to have these keys.
        return list(
            map(
                lambda p: QAPredictionDTO(
                    text=p["answer"],
                    score=p["score"],
                    start=p.get("start"),
                    end=p.get("end"),
                ),
                predictions,
            ),
        )


class DefaultQAModelWrapper(HFPipelineWrapper):
    """
    Deprecated: Use `HFPipelineWrapperForQuestionAnswering()`
    """

    def __init__(self, device: str = "cpu") -> None:
        raise DeprecationWarning(
            "Deprecated ModelWrapper. Please use `HFPipelineWrapperForQuestionAnswering`",
        )


def main():
    pass


if __name__ == "__main__":
    main()
