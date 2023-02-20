#!/usr/bin/env python3

from abc import abstractmethod
from typing import List

from jury import Jury

from .._base import AbstractBase
from ..misc.utils import format_to_jury
from ..structures import (
    EvaluationPredictionInstance,
    EvaluationReferenceInstance,
    MetricOutput,
)


class Metric(AbstractBase):
    """
    Metric is one of the components of the framework where the actual
    metric calculation happens.
    Any metric will compute a specific metric based on the predictions generated
    from any upstream language model.

    Any downstream implementation of `Metric` should implement the
    `compute(...)` method.

    > Note: An iterable of metric objects will be used for  `evaluators.Evaluator()`
    component.
    """

    @abstractmethod
    def compute(
        self,
        predictions: EvaluationPredictionInstance,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> MetricOutput:
        """
        The actual entrypoint method to perform evaluation and give output metric.

        Args:
            ```predictions```: ```EvaluationPredictionInstance```
                Could be:
                    - Single prediction `List[PredictionDTO]`
                    - Multiple prediction `List[List[PredictionDTO]]`
                Each prediction corresponds is one-to-one mapped to corresponding
                item in the `references` list.
                See  `evalem.structures` module to understand in detail.

            ```references```: ```EvaluationReferenceInstance```
                Could be:
                    - Single reference `List[ReferenceDTO]`
                    - Multiple reference `List[List[ReferenceDTO]]`
                Each reference corresponds is one-to-one mapped to corresponding
                item in the `predictions` list.
                See  `evalem.structures` module to understand in detail.

        Returns:
            Either a dictionary or a number after computing the metric.
        """
        raise NotImplementedError()

    def __call__(
        self,
        predictions: EvaluationPredictionInstance,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> MetricOutput:
        """
        The actual entrypoint method to perform evaluation and give output metric.

        Args:
            ```predictions```: ```EvaluationPredictionInstance```
                Could be:
                    - Single prediction `List[PredictionDTO]`
                    - Multiple prediction `List[List[PredictionDTO]]`
                Each prediction corresponds is one-to-one mapped to corresponding
                item in the `references` list.
                See  `evalem.structures` module to understand in detail.

            ```references```: ```EvaluationReferenceInstance```
                Could be:
                    - Single reference `List[ReferenceDTO]`
                    - Multiple reference `List[List[ReferenceDTO]]`
                Each reference corresponds is one-to-one mapped to corresponding
                item in the `predictions` list.
                See  `evalem.structures` module to understand in detail.

        Returns:
            Either a dictionary or a number after computing the metric.
        """
        return self.compute(predictions, references, **kwargs)


class JuryBasedMetric(Metric):
    """
    This is the metric component that's based on vanilla Jury scorer.

    Args:
        ```metrics```: ```Union[str, List[str]]```
            What metrics to compute?
        ```debug```: ```bool```
            Debug mode flag

    Some of the basic downstream implementation/inheritance using this are:
        - `metrics.basics.PrecisionMetric`
        - `metrics.basics.RecallMetric`
        - `metrics.basics.F1Metric`
        - `metrics.basics.AccuracyMetric`

    Direct usage:

        .. code-block: python

            from evalem.metrics._base import JuryBasedMetric

            references = [
                "Reference 1",
                "Reference 2"
            ]

            predictions = [
                PredictionDTO(text="Reference 1", score=1.0),
                PredictionDTO(text="Reference 2.5", score=0.75)
            ]

            # can use any of the `metrics.basics.BasicMetric`
            scorer = JuryBasedMetric(metrics=["f1", "precision"])
            result = scorer(predictions=predictions, references=references)
    """

    def __init__(self, metrics: List[str], debug: bool = False) -> None:
        """
        Args:
            ```metrics```: ```Union[str, List[str]]```
                What metrics to compute?
            ```debug```: ```bool```
                Debug mode flag
        """
        super().__init__(debug=debug)
        self.scorer = Jury(metrics=metrics)

    def compute(
        self,
        predictions: EvaluationPredictionInstance,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> MetricOutput:
        predictions = format_to_jury(predictions)
        references = format_to_jury(references)
        return self.scorer(predictions=predictions, references=references, **kwargs)


def main():
    pass


if __name__ == "__main__":
    main()
