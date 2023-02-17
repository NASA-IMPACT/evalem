#!/usr/bin/env python3

from typing import Dict, Iterable, List, Union

from transformers import pipeline as hf_pipeline

from ..structures import EvaluationPredictionInstance, QAPredictionDTO
from ._base import HFPipelineWrapper


class DefaultQAModelWrapper(HFPipelineWrapper):
    def __init__(self, debug: bool = False) -> None:
        super().__init__(pipeline=hf_pipeline("question-answering"))

    def _map_predictions(
        self,
        predictions: Union[dict, List[dict]],
    ) -> Iterable[EvaluationPredictionInstance]:
        if isinstance(predictions, dict):
            predictions = [predictions]
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
