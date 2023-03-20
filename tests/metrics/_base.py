# flake8: noqa
#!/usr/bin/env python3

import pytest

from .fixtures import predictions, references


class BaseMetricTest:
    _metric_cls = None
    _key = None

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
        assert isinstance(metric_result, dict)

    def test_metric_return_keys(self, metric_result):
        assert self._key in metric_result
        assert "score" in metric_result[self._key]


def main():
    pass


if __name__ == "__main__":
    main()
