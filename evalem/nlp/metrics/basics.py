#!/usr/bin/env python3

from ..._base.metrics import JuryBasedMetric
from ..._base.structures import (
    EvaluationReferenceInstance,
    MetricOutput,
    SinglePredictionInstance,
)
from ._base import NLPMetric


class ExactMatchMetric(JuryBasedMetric, NLPMetric):
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
