#!/usr/bin/env python3

from typing import Iterable

import pytest

from evalem.models import QuestionAnsweringHFPipelineWrapper
from evalem.structures import QAPredictionDTO

from .fixtures import squad_v2_data


@pytest.mark.models
class TestDefaultQAWrapper:
    @pytest.fixture(autouse=True, scope="class")
    def inputs(self, squad_v2_data):
        return squad_v2_data.get("inputs", [])

    @pytest.fixture(autouse=True, scope="class")
    def model(self):
        yield QuestionAnsweringHFPipelineWrapper(
            # model="distilbert-base-cased-distilled-squad"
        )

    @pytest.fixture(autouse=True, scope="class")
    def references(self, squad_v2_data):
        return squad_v2_data.get("references", [])

    @pytest.fixture(autouse=True, scope="class")
    def predictions(self, model, squad_v2_data):
        return model(squad_v2_data["inputs"])

    def test_predictions_format(self, predictions):
        assert isinstance(predictions, Iterable)
        assert isinstance(predictions[0], QAPredictionDTO)

    def test_predictions_len(self, predictions, references):
        print(predictions)
        assert len(predictions) == len(references)


def main():
    pass


if __name__ == "__main__":
    main()
