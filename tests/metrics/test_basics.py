# flake8: noqa
#!/usr/bin/env python3

import numpy as np

from evalem._base.metrics import (
    AccuracyMetric,
    ConfusionMatrix,
    F1Metric,
    PrecisionMetric,
    RecallMetric,
)

from ._base import BaseMetricTest, predictions, references


class TestAccuracyMetric(BaseMetricTest):
    _metric_cls = AccuracyMetric
    _key = "accuracy"

    def test_metric_score(self, metric_result):
        assert metric_result[self._key]["score"] >= 0


class TestF1Metric(BaseMetricTest):
    _metric_cls = F1Metric
    _key = "f1"

    def test_metric_score(self, metric_result):
        assert metric_result[self._key]["score"] >= 0


class TestPrecisionMetric(BaseMetricTest):
    _metric_cls = PrecisionMetric
    _key = "precision"

    def test_metric_score(self, metric_result):
        assert metric_result[self._key]["score"] >= 0


class TestRecallMetric(BaseMetricTest):
    _metric_cls = RecallMetric
    _key = "recall"

    def test_metric_score(self, metric_result):
        assert metric_result[self._key]["score"] >= 0


class TestConfusionMatrix(BaseMetricTest):
    _metric_cls = ConfusionMatrix
    _key = "confusion_matrix"

    def test_metric_return_keys(self, metric_result):
        assert self._key in metric_result

    def test_metric_score(self, metric_result):
        assert isinstance(metric_result[self._key], np.ndarray)
