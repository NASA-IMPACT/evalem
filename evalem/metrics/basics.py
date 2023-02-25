#!/usr/bin/env python3

from sklearn.metrics import confusion_matrix

from ..misc.utils import format_to_jury
from ..structures import (
    EvaluationReferenceInstance,
    MetricOutput,
    SinglePredictionInstance,
)
from ._base import JuryBasedMetric, Metric


class BasicMetric(Metric):
    """
    This represents generic metric implementation
    which is task-agnostic.
    Note:
        This exists only for the sake of type hierarchy.
    """

    pass


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


class ExactMatchMetric(JuryBasedMetric, BasicMetric):
    def __init__(self) -> None:
        super().__init__(metrics="exact_match")

    def compute(
        self,
        predictions: SinglePredictionInstance,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> MetricOutput:
        # This metric doesn't support multi-reference format.
        # So, we flatten everything:
        # Single Prediction, Multi-Ref -> Single Prediction, Single-Ref
        predictions, references = self._flatten_references(predictions, references)
        result = super().compute(
            predictions=predictions,
            references=references,
            **kwargs,
        )
        result["flattened"] = True
        return result


class ConfusionMatrix(BasicMetric):
    """
    This computes confusion matrix for the classification task.
    """

    def compute(
        self,
        predictions: SinglePredictionInstance,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> MetricOutput:
        # converts all the structure into list of string
        predictions, references = format_to_jury(predictions), format_to_jury(
            references,
        )

        predictions, references = self._flatten_references(predictions, references)

        labels = self.__get_labels(predictions, references)
        return dict(
            confusion_matrix=confusion_matrix(references, predictions, labels=labels),
            labels=labels,
            flattened=True,
            total_items=len(predictions),
            empty_items=0,
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
