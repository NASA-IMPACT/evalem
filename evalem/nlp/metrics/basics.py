#!/usr/bin/env python3

import dataclasses
from typing import Tuple

from ..._base.metrics import JuryBasedMetric
from ..._base.structures import (
    EvaluationPredictionInstance,
    EvaluationReferenceInstance,
    MetricResult,
    SinglePredictionInstance,
)
from ._base import NLPMetric


class ExactMatchMetric(JuryBasedMetric, NLPMetric):
    def __init__(self) -> None:
        super().__init__(metrics="exact_match")

    def _flatten_multi_prediction_multi_reference(
        self,
        predictions: EvaluationPredictionInstance,
        references: EvaluationReferenceInstance,
    ) -> Tuple[EvaluationPredictionInstance, EvaluationReferenceInstance]:
        raise NotImplementedError()

    def compute(
        self,
        predictions: SinglePredictionInstance,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> MetricResult:
        # This metric doesn't support multi-reference format.
        # So, we flatten everything:
        # Single Prediction, Multi-Ref -> Single Prediction, Single-Ref
        predictions, references = self._flatten_instances(predictions, references)
        result = super().compute(
            predictions=predictions,
            references=references,
            **kwargs,
        )

        extra = result.extra
        extra["flattened"] = True
        result = dataclasses.replace(
            result,
            score=result.extra.get("exact_match", None),
            extra=extra,
        )
        return result
