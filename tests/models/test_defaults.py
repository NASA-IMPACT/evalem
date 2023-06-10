#!/usr/bin/env python3

from typing import Iterable

import pytest

from evalem.nlp.structures import PredictionDTO


@pytest.mark.parametrize(
    "data, model",
    [
        ("squad_v2_data", "model_qa_default"),
        ("imdb_data", "model_classification_default"),
    ],
)
@pytest.mark.models
class TestDefaultModels:
    def test_predictions_format(self, data, model, request):
        data = request.getfixturevalue(data)
        model = request.getfixturevalue(model)
        predictions = model(data.get("inputs", []))
        assert isinstance(predictions, Iterable)
        assert isinstance(predictions[0], PredictionDTO)

    def test_predictions_len(self, data, model, request):
        data = request.getfixturevalue(data)
        model = request.getfixturevalue(model)
        predictions = model(data.get("inputs", []))
        print(predictions)
        assert len(predictions) == len(data.get("references", []))


def main():
    pass


if __name__ == "__main__":
    main()
