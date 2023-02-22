#!/usr/bin/env python3

from typing import Iterable, List, Union

from transformers import pipeline as hf_pipeline

from ..structures import EvaluationPredictionInstance, QAPredictionDTO
from ._base import HFPipelineWrapper


class DefaultQAModelWrapper(HFPipelineWrapper):
    """
    A default distill-bert-uncased base HF pipeline for
    Question-Answering task.
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
