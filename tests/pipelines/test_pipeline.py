#!/usr/bin/env python3

from pprint import pprint
from typing import Iterable

import pytest

from evalem.evaluators import QAEvaluator, TextClassificationEvaluator
from evalem.models import (
    QuestionAnsweringHFPipelineWrapper,
    TextClassificationHFPipelineWrapper,
)
from evalem.pipelines import SimpleEvaluationPipeline

from ..models.fixtures import imdb_data, squad_v2_data
from ..models.test_classification import TestDefaultTextClassificationWrapper
from ..models.test_qa import TestDefaultQAWrapper


@pytest.mark.dependency(depends=["TestDefaultQAWrapper"])
@pytest.mark.pipelines
class TestSimpleEvaluationPipelineForQA:
    @pytest.fixture(autouse=True, scope="class")
    def inputs(self, squad_v2_data):
        return squad_v2_data.get("inputs", [])

    @pytest.fixture(autouse=True, scope="class")
    def model(self):
        yield QuestionAnsweringHFPipelineWrapper()

    @pytest.fixture(autouse=True, scope="class")
    def evaluators(self):
        yield QAEvaluator()

    @pytest.fixture(autouse=True, scope="class")
    def references(self, squad_v2_data):
        return squad_v2_data.get("references", [])

    def test_evaluation(self, model, evaluators, inputs, references):
        pipeline = SimpleEvaluationPipeline(model=model, evaluators=evaluators)
        results = pipeline(inputs, references)
        assert isinstance(results, Iterable)


@pytest.mark.dependency(depends=["TestDefaultTextClassificationWrapper"])
@pytest.mark.pipelines
class TestSimpleEvaluationPipelineForTextClassification:
    @pytest.fixture(autouse=True, scope="class")
    def inputs(self, imdb_data):
        return imdb_data.get("inputs", [])

    @pytest.fixture(autouse=True, scope="class")
    def model(self):
        yield TextClassificationHFPipelineWrapper(hf_params=dict(truncation=True))

    @pytest.fixture(autouse=True, scope="class")
    def evaluators(self):
        yield TextClassificationEvaluator()

    @pytest.fixture(autouse=True, scope="class")
    def references(self, imdb_data):
        return imdb_data.get("references", [])

    def test_evaluation(self, model, evaluators, inputs, references):
        pipeline = SimpleEvaluationPipeline(model=model, evaluators=evaluators)
        results = pipeline(inputs, references)
        assert isinstance(results, Iterable)


def main():
    pass


if __name__ == "__main__":
    main()
