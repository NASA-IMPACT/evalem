#!/usr/bin/env python3
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Type, Union

from .abc import AbstractBase
from ..metrics import Metric
from ..metrics.basics import AccuracyMetric
from .structures import (
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

    Args:
        ```metrics```: ```Optional[Iterable[Type[Metric]]]```
            An iterable (list/set/tuple) of metric objects.
            - If None, default `Evaluator.DEFAULT_METRIC_CLS` is used

    Direct usage:
            .. code-block: python

                from evalem.evaluators import Evaluator

                # could be List of text
                references = [
                    "Reference 1",
                    "Reference 2"
                ]

                # Or list of ReferenceDTO
                references = [
                    ReferenceDTO(text="Reference 1"),
                    ReferenceDTO(text="Reference 2")
                ]

                # or list of dict
                referencs = [
                    dict(text="Reference 1"),
                    dict(text="Reference 2"),
                ]

                # similarly for predictions
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

    # default is single accuracy metric
    DEFAULT_METRIC_CLS = AccuracyMetric

    def __init__(
        self,
        metrics: Optional[Iterable[Type[Metric]]] = None,
        debug: bool = False,
    ) -> None:
        super().__init__(debug)

        metrics = list(metrics or [Evaluator.DEFAULT_METRIC_CLS()])
        self._type_check_metrics(metrics)
        self.metrics = metrics

    @staticmethod
    def _type_check_metrics(
        metrics: Union[Type[Metric], Iterable[Type[Metric]]],
    ) -> bool:
        """
        Validates the type of metric list or just single metric object.

        Args:
            ```metrics``` : ```Union[Type[Metric], Iterable[Type[Metric]]]```
                Input single `Metric` object or an iterable of objects

        Returns:
            Returns True if everything is okay. Otherwise throws a type error.

        """
        metrics = [metrics] if not isinstance(metrics, Iterable) else metrics
        for _metric in metrics:
            if not isinstance(_metric, Metric):
                print(_metric)
                raise TypeError(
                    f"Invalid type for metric={_metric}. Expected type of [Metric]. Got {type(_metric)}",
                )
        return True

    def add_metric(self, metric: Type[Metric]) -> Type[Evaluator]:
        """
        A public interface to add metric to the existing list.
        Uses the builder-pattern to add and build the object.
        """
        self._type_check_metrics(metric)
        self.metrics.append(metric)
        return self

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
