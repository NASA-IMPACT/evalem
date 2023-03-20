#!/usr/bin/env python3

from pprint import pprint
from typing import Iterable

import pytest

from evalem.models import TextClassificationHFPipelineWrapper
from evalem.structures import PredictionDTO

from .fixtures import imdb_data


@pytest.mark.models
class TestDefaultTextClassificationWrapper:
    @pytest.fixture(autouse=True, scope="class")
    def inputs(self, imdb_data):
        return imdb_data.get("inputs", [])

    @pytest.fixture(autouse=True, scope="class")
    def model(self):
        yield TextClassificationHFPipelineWrapper(hf_params=dict(truncation=True))

    @pytest.fixture(autouse=True, scope="class")
    def references(self, imdb_data):
        return imdb_data.get("references", [])

    @pytest.fixture(autouse=True, scope="class")
    def predictions(self, model, imdb_data):
        return model(imdb_data["inputs"])

    def test_predictions_format(self, predictions):
        assert isinstance(predictions, Iterable)
        assert isinstance(predictions[0], PredictionDTO)

    def test_predictions_len(self, predictions, references):
        pprint(f"Predictions | {predictions}")
        assert len(predictions) == len(references)


def main():
    pass


if __name__ == "__main__":
    main()
