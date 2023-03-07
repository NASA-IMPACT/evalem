#!/usr/bin/env python3

from evalem.metrics import BertScore

from ._base import BaseMetricTest, predictions, references


class TestBertScore(BaseMetricTest):
    _metric_cls = BertScore
    _key = "bertscore"
