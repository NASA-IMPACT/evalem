#!/usr/bin/env python3

from ..._base.metrics import (
    AccuracyMetric,
    ConfusionMatrix,
    F1Metric,
    PrecisionMetric,
    RecallMetric,
)
from ..metrics import ExactMatchMetric
from ._base import NLPEvaluator


class QAEvaluator(NLPEvaluator):
    """
    An evaluator for QA-based tasks.
    """

    def __init__(self) -> None:
        super().__init__(
            metrics=[
                AccuracyMetric(),
                ExactMatchMetric(),
                F1Metric(),
            ],
        )


class TextClassificationEvaluator(NLPEvaluator):
    """
    An evaluator for text classification tasks.
    """

    def __init__(self) -> None:
        super().__init__(
            metrics=[
                AccuracyMetric(),
                F1Metric(),
                PrecisionMetric(),
                RecallMetric(),
                ConfusionMatrix(),
            ],
        )


def main():
    pass


if __name__ == "__main__":
    main()
