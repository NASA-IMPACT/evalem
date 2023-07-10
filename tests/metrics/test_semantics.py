# flake8: noqa
#!/usr/bin/env python3

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
    _key = "bertscore"

    def test_metric_score(self, metric_result):
        assert -1 <= metric_result["score"] <= 1


class TestBartScore(BaseMetricTest):
    _metric_cls = BartScore
    _key = "bartscore"

    def test_metric_score(self, metric_result):
        assert -10 <= metric_result["score"] <= 10


class TestBleuMetric(BaseMetricTest):
    _metric_cls = BleuMetric
    _key = "bleu"

    def test_metric_score(self, metric_result):
        pprint(metric_result)
        assert 0 <= metric_result["score"] <= 1


class TestSacreBleuMetric(BaseMetricTest):
    _metric_cls = SacreBleuMetric
    _key = "sacrebleu"

    def test_metric_score(self, metric_result):
        pprint(metric_result)
        assert 0 <= metric_result["score"] <= 1


class TestMeteorMetric(BaseMetricTest):
    _metric_cls = MeteorMetric
    _key = "meteor"

    def test_metric_score(self, metric_result):
        pprint(metric_result)
        assert 0 <= metric_result["score"] <= 1


class TestRougeMetric(BaseMetricTest):
    _metric_cls = RougeMetric
    _key = "rouge"

    def test_metric_return_keys(self, metric_result):
        assert self._key in metric_result
        assert "rouge1" in metric_result[self._key]
        assert "rouge2" in metric_result[self._key]
        assert "rougeL" in metric_result[self._key]
        assert "rougeLsum" in metric_result[self._key]

    def test_metric_score(self, metric_result):
        pprint(metric_result)
        assert 0 <= metric_result[self._key]["rouge1"] <= 1
        assert 0 <= metric_result[self._key]["rouge2"] <= 1
        assert 0 <= metric_result[self._key]["rougeL"] <= 1
        assert 0 <= metric_result[self._key]["rougeLsum"] <= 1
