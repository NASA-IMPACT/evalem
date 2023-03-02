#!/usr/bin/env python3

from typing import Iterable, List, Union

from transformers import pipeline as hf_pipeline

from ..structures import EvaluationPredictionInstance, QAPredictionDTO
from ._base import HFPipelineWrapper


class DefaultQAModelWrapper(HFPipelineWrapper):
    """
    A default distill-bert-uncased base HF pipeline for
    Question-Answering task.

    The predictor expects the input format to be a `List[dict]`, where each
    dict has the following keys:
        - `context` (str): Paragraph/context fromw which question is asked
        - `question` (str): Actual question string being asked

    Example input dict:
            .. code-block: python

                {
                    "context": "There are 7 continents in the world."
                    "question": "How many continents are there?"
                }

    The `predict(...)` method finally returns `List[QAPredictionDTO]` structure.
    """

    def __init__(self, device: str = "cpu") -> None:
        super().__init__(pipeline=hf_pipeline("question-answering", device=device))

    def _map_predictions(
        self,
        predictions: Union[dict, List[dict]],
    ) -> Iterable[EvaluationPredictionInstance]:
        """
        This helper method converts the pipeline's default output format
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


def main():
    pass


if __name__ == "__main__":
    main()
