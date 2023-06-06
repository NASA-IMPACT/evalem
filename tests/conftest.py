#!/usr/bin/env python3
import datasets
import pytest

from evalem.evaluators import QAEvaluator, TextClassificationEvaluator
from evalem.models import (
    QuestionAnsweringHFPipelineWrapper,
    TextClassificationHFPipelineWrapper,
)


@pytest.fixture(autouse=True, scope="session")
def squad_v2_data():
    _data = datasets.load_dataset(
        path="squad_v2",
        split=datasets.ReadInstruction(split_name="validation", from_=0, to=10),
    )
    data = _data.to_dict()
    return data


@pytest.fixture(autouse=True, scope="session")
def imdb_data():
    _data = datasets.load_dataset(
        path="imdb",
        split=datasets.ReadInstruction(split_name="test", from_=0, to=10),
    )
    data = _data.to_dict()
    return data


@pytest.fixture(autouse=True, scope="session")
def model_qa_default():
    yield QuestionAnsweringHFPipelineWrapper()


@pytest.fixture(autouse=True, scope="session")
def model_classification_default():
    yield TextClassificationHFPipelineWrapper(hf_params=dict(truncation=True))


@pytest.fixture(autouse=True, scope="session")
def evaluator_qa_default():
    yield QAEvaluator()


@pytest.fixture(autouse=True, scope="session")
def evaluator_classification_default():
    yield TextClassificationEvaluator()


def main():
    pass


if __name__ == "__main__":
    main()
