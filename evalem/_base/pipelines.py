#!/usr/bin/env python3

from abc import abstractmethod
from typing import Any, Iterable, List, Mapping, Type, Union

from .abc import AbstractBase
from .evaluators import Evaluator
from .models import ModelWrapper
from .structures import EvaluationReferenceInstance, MetricOutput


class EvaluationPipeline(AbstractBase):
    """
    Represents a type for Pipeline component.
    All the downstream pipeline object should implement the `.run(...)` method.

    See `pipelines.defaults.SimpleEvaluationPipeline` for an implementation.
    """

    @abstractmethod
    def run(self, *args, **kwags) -> Any:
        """
        Entry-point method to run the evaluation.
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> Any:
        return self.run(*args, **kwargs)


class SimpleEvaluationPipeline(EvaluationPipeline):
    """
    This is a very basic evaluation pipeline that uses single model
    and a list of evaluators to run the evaluation.

    Args:
        ```model```: ```Type[ModelWrapper]```
            Wrapped model to do the inference.
        ```evaluators```: ```Union[Evaluator, Iterable[Evalautor]]```
            Either a single evaluator or an iterable of evaluators
            Note: If single evaluator is provided, it'll be wrapped into
            an iterable ultimately.

    Usage:

        .. code-block: python

            from evalem.pipelines import SimpleEvaluationPipeline
            from evalem.models import TextClassificationHFPipelineWrapper
            from evalem.evaluators import TextClassificationEvaluator

            model = TextClassificationHFPipelineWrapper()
            evaluator = TextClassificationEvaluator()
            pipe = SimpleEvaluationPipeline(model=model, evaluators=evaluator)

            results = pipe(inputs, references)

            # or
            results = pipe.run(inputs, references)
    """

    def __init__(
        self,
        model: Type[ModelWrapper],
        evaluators: Union[Evaluator, Iterable[Evaluator]],
    ) -> None:
        self.model = model

        # if only single evaluator, wrap into an iterable
        self.evaluators = (
            [evaluators] if not isinstance(evaluators, Iterable) else evaluators
        )

    def run(
        self,
        inputs: Mapping,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> List[MetricOutput]:
        """
        ```inputs```: ```Mapping```
            Input data to run over the model to get predictions.
        ```references```: ```EvaluationReferenceInstance```
            References/ground-truths to be used for evaluation.
            See `evalem.metrics`   for more information.
        """
        predictions = self.model(inputs, **kwargs.get("model_params", {}))
        return list(
            map(
                lambda e: e(
                    predictions=predictions,
                    references=references,
                    **kwargs.get("eval_params", {}),
                ),
                self.evaluators,
            ),
        )


def main():
    pass


if __name__ == "__main__":
    main()