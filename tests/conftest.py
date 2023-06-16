#!/usr/bin/env python3
import pytest

from evalem.nlp.evaluators import QAEvaluator, TextClassificationEvaluator
from evalem.nlp.misc.datasets import get_imdb, get_squad_v2
from evalem.nlp.models import (

    QuestionAnsweringHFPipelineWrapper,
    TextClassificationHFPipelineWrapper,
)


@pytest.fixture(autouse=True, scope="session")
def squad_v2_data():
    data = get_squad_v2(data_type="validation", nsamples=10, shuffle=False)
    return data


@pytest.fixture(autouse=True, scope="session")
def imdb_data():
    data = get_imdb(data_type="test", nsamples=10, shuffle=False)
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
