#!/usr/bin/env python3

from .._base.metrics import (
    EvaluationPredictionInstance,
    EvaluationReferenceInstance,
    Metric,
    MetricOutput,
)


class NullMetric(Metric):
    """
    A dummy metric computer that's irrelevant and only used for testing purposes.
    """

    def compute(
        self,
        predictions: EvaluationPredictionInstance,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> MetricOutput:
        return 0.0


def main():
    pass


if __name__ == "__main__":
    main()
