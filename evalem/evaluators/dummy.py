#!/usr/bin/env python3

import os
import sys

from ._base import (
    EvaluationOutput,
    EvaluationPredictionInstance,
    EvaluationReferenceInstance,
    Evaluator,
)


class NullEvaluator(Evaluator):
    """
    A dummy evaluator that's irrelevant and only used for testing purposes.
    """

    def evaluate(
        self,
        predictions: EvaluationPredictionInstance,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> EvaluationOutput:
        return 0.0


def main():
    pass


if __name__ == "__main__":
    main()
