#!/usr/bin/env python3

from ..metrics import AccuracyMetric, ExactMatchMetric, F1Metric
from ._base import Evaluator


class BasicEvaluator(Evaluator):
    """
    This represents generic evaluator implementation
    which is task-agnostic.
    Note:
        This exists only for the sake of type hierarchy.
    """

    pass


class QAEvaluator(BasicEvaluator):
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


def main():
    pass


if __name__ == "__main__":
    main()
