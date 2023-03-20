# flake8: noqa
#!/usr/bin/env python3

from evalem.metrics import BartScore, BertScore

from ._base import BaseMetricTest, predictions, references


class TestBertScore(BaseMetricTest):
    _metric_cls = BertScore
    _key = "bertscore"

    def test_metric_score(self, metric_result):
        assert -1 <= metric_result[self._key]["score"] <= 1


class TestBartScore(BaseMetricTest):
    _metric_cls = BartScore
    _key = "bartscore"

    def test_metric_score(self, metric_result):
        assert -10 <= metric_result[self._key]["score"] <= 10
