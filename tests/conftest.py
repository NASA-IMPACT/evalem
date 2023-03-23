#!/usr/bin/env python3

import json
from pathlib import Path

import pytest

from evalem.evaluators import QAEvaluator, TextClassificationEvaluator
from evalem.models import (
    QuestionAnsweringHFPipelineWrapper,
    TextClassificationHFPipelineWrapper,
)


@pytest.fixture(autouse=True, scope="session")
def squad_v2_data():
    path = Path(__file__).parent.joinpath("data/squad_v2.json")
    data = {}
    with open(path, "r") as f:
        data = json.load(f)
    return data


@pytest.fixture(autouse=True, scope="session")
def imdb_data():
    path = Path(__file__).parent.joinpath("data/imdb.json")
    data = {}
    with open(path, "r") as f:
        data = json.load(f)
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
