#!/usr/bin/env python3

from pprint import pprint
from typing import Iterable

import pytest

from evalem.pipelines import SimpleEvaluationPipeline

# from ..models.test_defaults import TestDefaultModels


@pytest.mark.dependency(depends=["TestDefaultModels"])
@pytest.mark.parametrize(
    "data, model, evaluators",
    [
        ("squad_v2_data", "model_qa_default", "evaluator_qa_default"),
        (
            "imdb_data",
            "model_classification_default",
            "evaluator_classification_default",
        ),
    ],
)
@pytest.mark.pipelines
class TestSimplePipeline:
    def test_evaluation(self, data, model, evaluators, request):
        data = request.getfixturevalue(data)
        model = request.getfixturevalue(model)
        evaluators = request.getfixturevalue(evaluators)
        pipeline = SimpleEvaluationPipeline(model=model, evaluators=evaluators)

        inputs, references = data.get("inputs", []), data.get("references", [])

        results = pipeline(inputs, references)
        pprint(results)

        assert isinstance(results, Iterable)


def main():
    pass


if __name__ == "__main__":
    main()
