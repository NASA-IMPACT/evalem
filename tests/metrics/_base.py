#!/usr/bin/env python3

import pytest

from .fixtures import predictions, references


class BaseMetricTest:
    _metric_cls = None
    _key = None

    @pytest.fixture(autouse=True)
    def metric_result(self, predictions, references):
        return self._metric_cls()(predictions=predictions, references=references)

    def test_predictions_references_len(self, predictions, references):
        assert len(predictions) == len(references)

    def test_metric_return_type(self, metric_result):
        assert isinstance(metric_result, dict)

    def test_metric_return_keys(self, metric_result):
        assert self._key in metric_result
        assert "score" in metric_result[self._key]


def main():
    pass


if __name__ == "__main__":
    main()
