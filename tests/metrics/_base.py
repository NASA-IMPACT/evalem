# flake8: noqa
#!/usr/bin/env python3

from pprint import pprint

import pytest

from evalem._base.structures import MetricResult

from .fixtures import predictions, references


@pytest.mark.metrics
class BaseMetricTest:
    _metric_cls = None

    @pytest.fixture(autouse=True, scope="class")
    def metric_result(self, predictions, references):
        """
        This is used to cache the computation per class
        """
        return self._metric_cls()(predictions=predictions, references=references)

    def test_predictions_references_len(self, predictions, references):
        """
        Test the length of input predictions and references
        """
        assert len(predictions) == len(references)

    def test_metric_return_type(self, metric_result):
        """
        Check if return type of each metric is a dictionary
        """
        assert isinstance(metric_result, MetricResult)

    def test_metric_score(self, metric_result):
        pprint(metric_result)
        assert 0 <= metric_result.score <= 1


def main():
    pass


if __name__ == "__main__":
    main()
