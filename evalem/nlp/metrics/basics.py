#!/usr/bin/env python3

import dataclasses
import string

from ..._base.metrics import JuryBasedMetric
from ..._base.structures import (
    EvaluationReferenceInstance,
    MetricResult,
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
    ) -> MetricResult:
        # This metric doesn't support multi-reference format.
        # So, we flatten everything:
        # Single Prediction, Multi-Ref -> Single Prediction, Single-Ref
        predictions, references = self._flatten_references(predictions, references)
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


class PartialMatchMetric(NLPMetric):
    """
    This metric is used when we want to check if a text lies in another text.
    It's robust than ExactMatchMetric and gives better matching score.
    It's a match if:
        - prediction string lies in the corresponding reference string
        - reference string lies in the corresponding prediction string
    """

    def __init__(self, preprocess: bool = True, debug: bool = False) -> None:
        super().__init__(debug=debug)
        self.preprocess = bool(preprocess)

    def compute(
        self,
        predictions: SinglePredictionInstance,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> MetricResult:
        predictions, references = self._flatten_references(predictions, references)
        if self.preprocess:
            _prep = lambda x: x.lower().strip(string.punctuation).strip()
            references = list(map(_prep, references))
            predictions = list(map(_prep, predictions))
        matches = 0
        for ref, pred in zip(references, predictions):
            if (pred in ref) or (ref in pred):
                matches += 1

        total_items = len(predictions)
        score = matches / total_items if total_items else 0.0
        return MetricResult(
            score=score,
            total_items=total_items,
            metric_name=self.__classname__,
            extra=dict(flattened=True),
        )
