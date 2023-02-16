#!/usr/bin/env python3

from jury import Jury

from ._base import (
    EvaluationOutput,
    EvaluationPredictionInstance,
    EvaluationReferenceInstance,
    Evaluator,
    JuryBasedEvaluator,
)


class BasicEvaluator(Evaluator):
    """
    This represents generic evaluator implementation
    which is task-agnostic.
    Note:
        This exists only for the sake of type hierarchy.
    """

    pass


class PrecisionEvaluator(JuryBasedEvaluator, BasicEvaluator):
    def __init__(self) -> None:
        super().__init__(metrics="precision")


class RecallEvaluator(JuryBasedEvaluator, BasicEvaluator):
    def __init__(self) -> None:
        super().__init__(metrics="recall")


class F1Evaluator(JuryBasedEvaluator, BasicEvaluator):
    def __init__(self) -> None:
        super().__init__(metrics="f1")


class AccuracyEvaluator(JuryBasedEvaluator, BasicEvaluator):
    def __init__(self) -> None:
        super().__init__(metrics="accuracy")


def main():
    pass


if __name__ == "__main__":
    main()
