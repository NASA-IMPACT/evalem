# flake8: noqa
#!/usr/bin/env python3

import math
from pprint import pprint

from evalem.nlp.metrics import (
    BartScore,
    BertScore,
    BleuMetric,
    MeteorMetric,
    RougeMetric,
    SacreBleuMetric,
)

from ._base import BaseMetricTest, predictions, references


class TestBertScore(BaseMetricTest):
    _metric_cls = BertScore

    def test_metric_score(self, metric_result):
        assert -1 <= metric_result.score <= 1


class TestBartScore(BaseMetricTest):
    _metric_cls = BartScore

    def test_metric_score(self, metric_result):
        assert -math.inf <= metric_result.score <= math.inf


class TestBleuMetric(BaseMetricTest):
    _metric_cls = BleuMetric

    def test_metric_score(self, metric_result):
        pprint(metric_result)
        assert 0 <= metric_result.score <= 1


class TestSacreBleuMetric(BaseMetricTest):
    _metric_cls = SacreBleuMetric

    def test_metric_score(self, metric_result):
        pprint(metric_result)
        assert 0 <= metric_result.score <= 1


class TestMeteorMetric(BaseMetricTest):
    _metric_cls = MeteorMetric

    def test_metric_score(self, metric_result):
        pprint(metric_result)
        assert 0 <= metric_result.score <= 1


class TestRougeMetric(BaseMetricTest):
    _metric_cls = RougeMetric

    def test_metric_return_keys(self, metric_result):
        key = "rouge"
        assert key in metric_result.extra
        assert "rouge1" in metric_result.extra[key]
        assert "rouge2" in metric_result.extra[key]
        assert "rougeL" in metric_result.extra[key]
        assert "rougeLsum" in metric_result.extra[key]

    def test_metric_score(self, metric_result):
        key = "rouge"
        pprint(metric_result)
        assert 0 <= metric_result.extra[key]["rouge1"] <= 1
        assert 0 <= metric_result.extra[key]["rouge2"] <= 1
        assert 0 <= metric_result.extra[key]["rougeL"] <= 1
        assert 0 <= metric_result.extra[key]["rougeLsum"] <= 1
