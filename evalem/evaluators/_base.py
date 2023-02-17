#!/usr/bin/env python3
from __future__ import annotations

from abc import abstractmethod
from typing import Iterable, Mapping, Type

from .._base import AbstractBase
from ..metrics import Metric
from ..structures import (
    EvaluationOutput,
    EvaluationPredictionInstance,
    EvaluationReferenceInstance,
)


class Evaluator(AbstractBase):
    """
    Evaluator is one of the components of the framework where the actual evaluation happens.
    Any evaluators will compute a specific metric based on the predictions generated
    from any upstream language model.

    Any downstream implementation of `Evaluator` should implement the
    `evaluate(...)` method.

    Each `Evaluator` type consists of one or more `metrics.Metric` object.
    These objects can be passed via:
        - constructor `Evaluator(metrics=[...])`
        - `add_metric(<metric_object>)`

    Direct usage:
            .. code-block: python

                from evalem.evaluators import Evaluator
                references = [
                    "Reference 1",
                    "Reference 2"
                ]
                predictions = [
                    PredictionDTO(text="Reference 1", score=1.0),
                    PredictionDTO(text="Reference 2.5", score=0.75)
                ]

                # create evaluator
                evaluator = Evaluator(metrics=[
                    NullMetric(),
                    PrecisionMetric(),
                    RecallMetric(),
                    F1Metric(),
                    AccuracyMetric()
                ])

                # or builder pattern
                evaluator = (
                    Evaluator([])
                    .add_metric(NullMetric())
                    .add_metric(PrecisionMetric())
                    .add_metric(RecallMetric())
                    .add_metric(F1Metric())
                    .add_metric(AccuracyMetric())
                )

                result = evaluator(predictions=predictions, references=references)
    """

    def __init__(self, metrics: Iterable[Type[Metric]], debug: bool = False) -> None:
        super().__init__(debug)
        self.metrics = list(metrics)

    def add_metric(self, metric: Type[Metric]) -> Type[Evaluator]:
        self.metrics.append(metric)
        return self

    def __getitem__(self, index):
        return self.metrics[index]

    def evaluate(
        self,
        predictions: EvaluationPredictionInstance,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> Mapping[str, dict]:
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
            Mapping (dict) of metric name to corresponding metric output
        """
        return dict(
            map(
                lambda m: (
                    m.__classname__,
                    m(predictions=predictions, references=references, **kwargs),
                ),
                self.metrics,
            ),
        )

    def __call__(
        self,
        predictions: EvaluationPredictionInstance,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> EvaluationOutput:
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
            Mapping (dict) of metric name to corresponding metric output
        """
        return self.evaluate(predictions, references, **kwargs)

    def __repr__(self) -> str:
        metric_str = "".join(map(str, self.metrics))
        return f"{super().__repr__()} || {metric_str}"


def main():
    pass


if __name__ == "__main__":
    main()
