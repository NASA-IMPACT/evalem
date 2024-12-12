#!/usr/bin/env python3

from __future__ import annotations

from abc import abstractmethod
from typing import Iterable, List, Tuple

from jury import Jury
from sklearn.metrics import confusion_matrix

from ..misc.utils import format_to_jury
from .abc import AbstractBase
from .structures import (
    EvaluationPredictionInstance,
    EvaluationReferenceInstance,
    MetricResult,
    SequenceType,
    SinglePredictionInstance,
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

    def __init__(self, device: str = "cpu", debug: bool = False) -> None:
        super().__init__(debug=debug)
        self.device = device

    @abstractmethod
    def compute(
        self,
        predictions: EvaluationPredictionInstance,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> MetricResult:
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
    ) -> MetricResult:
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
        return self.compute(
            predictions=predictions,
            references=references,
            **kwargs,
        )

    @staticmethod
    def _flatten_references(
        predictions: EvaluationPredictionInstance,
        references: EvaluationReferenceInstance,
    ) -> Tuple[EvaluationPredictionInstance, EvaluationReferenceInstance]:
        """
        This flattens the nested formats:
            From Single Prediction, Multiple References to
            Single Prediction, Single Reference
            (See `metrics.basics.ExactMatchMetric` for usage)

        Args:
            ```predictions```: ```EvaluationPredictionInstance```
                Input list of predictions
            ```predictions```: ```EvaluationPredictionInstance```
                Input list of references

        Returns:
            Tuple of flattened lists (predictions, references)
        """
        res = []
        for pred, ref in zip(predictions, references):
            # if multiple predictions, skip for now
            if isinstance(pred, SequenceType) and not isinstance(pred, str):
                raise TypeError("Cannot handle multiple prediction instance")
            # if multiple references
            elif isinstance(ref, SequenceType) and not isinstance(ref, str):
                res.extend(list(map(lambda r: (pred, r), ref)))
            else:
                res.append((pred, ref))
        predictions, references = zip(*res)
        return predictions, references


class BasicMetric(Metric):
    """
    This represents generic metric implementation
    which is task-agnostic.
    Note:
        This exists only for the sake of type hierarchy.
    """

    pass


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

    Note:
        We create some basic metrics for any nlp/cv tasks to be used.
        So, all the input predictions/references as labels are converted to strings.

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

    def __init__(
        self,
        metrics: List[str],
        device: str = "cpu",
        debug: bool = False,
    ) -> None:
        """
        Args:
            ```metrics```: ```Union[str, List[str]]```
                What metrics to compute?
            ```debug```: ```bool```
                Debug mode flag
        """
        super().__init__(device=device, debug=debug)
        self.scorer = Jury(metrics=metrics)

    def compute(
        self,
        predictions: EvaluationPredictionInstance,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> MetricResult:
        predictions = format_to_jury(predictions)
        references = format_to_jury(references)

        results = self.scorer(
            predictions=predictions,
            references=references,
            **kwargs,
        )
        res = dict()
        for k, v in results.items():
            # for single metrics, just flatten the dict that has "score" key
            if isinstance(v, dict) and "score" in v:
                res["score"] = v.get("score", None)
            res[k] = v
        res["metric_name"] = self.__classname__
        return MetricResult.from_dict(res)


class PrecisionMetric(JuryBasedMetric, BasicMetric):
    def __init__(self) -> None:
        super().__init__(metrics="precision")


class RecallMetric(JuryBasedMetric, BasicMetric):
    def __init__(self) -> None:
        super().__init__(metrics="recall")


class F1Metric(JuryBasedMetric, BasicMetric):
    def __init__(self) -> None:
        super().__init__(metrics="f1")


class AccuracyMetric(JuryBasedMetric, BasicMetric):
    def __init__(self) -> None:
        super().__init__(metrics="accuracy")


class ConfusionMatrix(BasicMetric):
    """
    This computes confusion matrix for the classification task.
    """

    def compute(
        self,
        predictions: EvaluationPredictionInstance,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> MetricResult:
        # converts all the structure into list of string
        predictions, references = format_to_jury(predictions), format_to_jury(
            references,
        )

        predictions, references = self._flatten_references(predictions, references)

        labels = self.__get_labels(predictions, references)
        return MetricResult.from_dict(
            dict(
                metric_name="ConfusionMatrix",
                confusion_matrix=confusion_matrix(
                    references,
                    predictions,
                    labels=labels,
                ),
                labels=labels,
                flattened=True,
                total_items=len(predictions),
                empty_items=0,
            ),
        )

    def __get_labels(
        self,
        predictions: SinglePredictionInstance,
        references: SinglePredictionInstance,
    ):
        """
        Get unique list of labels across predictions + references.
        """
        return sorted(set(predictions).union(references))


def main():
    pass


if __name__ == "__main__":
    main()
