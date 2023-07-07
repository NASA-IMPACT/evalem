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

    def test_metric_score(self, metric_result):
        assert metric_result.score >= 0


class TestF1Metric(BaseMetricTest):
    _metric_cls = F1Metric

    def test_metric_score(self, metric_result):
        assert metric_result.score >= 0


class TestPrecisionMetric(BaseMetricTest):
    _metric_cls = PrecisionMetric

    def test_metric_score(self, metric_result):
        assert metric_result.score >= 0


class TestRecallMetric(BaseMetricTest):
    _metric_cls = RecallMetric

    def test_metric_score(self, metric_result):
        assert metric_result.score >= 0


class TestConfusionMatrix(BaseMetricTest):
    _metric_cls = ConfusionMatrix

    def test_metric_score(self, metric_result):
        assert isinstance(metric_result.extra["confusion_matrix"], np.ndarray)
