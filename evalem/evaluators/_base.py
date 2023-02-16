#!/usr/bin/env python3

from abc import abstractmethod
from typing import List

from jury import Jury

from .._base import AbstractBase
from ..misc.utils import format_to_jury
from ..structures import (
    EvaluationOutput,
    EvaluationPredictionInstance,
    EvaluationReferenceInstance,
)


class Evaluator(AbstractBase):
    """
    Evaluator is one of the components of the framework where the actual evaluation happens.
    Any evaluators will compute a specific metric based on the predictions generated
    from any upstream language model.

    Any downstream implementation of `Evaluator` should implement the
    `evaluate(...)` method.
    """

    @abstractmethod
    def evaluate(
        self,
        predictions: EvaluationPredictionInstance,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> EvaluationOutput:
        """
        The actual entrypoint method to perform evaluation and give output metric.

        Args:
            ```predictions```: ```EvaluationPredictionInstance```
                Could be:
                    - Single prediction `List[PredictionDTO]`
                    - Multiple prediction `List[List[PredictionDTO]]`
                Each prediction corresponds is one-to-one mapped to corresponding
                item in the `references` list.
                See  `evalem.structures` module to understand in detail.

            ```references```: ```EvaluationReferenceInstance```
                Could be:
                    - Single reference `List[ReferenceDTO]`
                    - Multiple reference `List[List[ReferenceDTO]]`
                Each reference corresponds is one-to-one mapped to corresponding
                item in the `predictions` list.
                See  `evalem.structures` module to understand in detail.

        Returns:
            Either a dictionary or a number after computing the metric.
        """
        raise NotImplementedError()
        pass

    def __call__(
        self,
        predictions: EvaluationPredictionInstance,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> EvaluationOutput:
        """
        The actual entrypoint method to perform evaluation and give output metric.

        Args:
            ```predictions```: ```EvaluationPredictionInstance```
                Could be:
                    - Single prediction `List[PredictionDTO]`
                    - Multiple prediction `List[List[PredictionDTO]]`
                Each prediction corresponds is one-to-one mapped to corresponding
                item in the `references` list.
                See  `evalem.structures` module to understand in detail.

            ```references```: ```EvaluationReferenceInstance```
                Could be:
                    - Single reference `List[ReferenceDTO]`
                    - Multiple reference `List[List[ReferenceDTO]]`
                Each reference corresponds is one-to-one mapped to corresponding
                item in the `predictions` list.
                See  `evalem.structures` module to understand in detail.

        Returns:
            Either a dictionary or a number after computing the metric.
        """
        return self.evaluate(predictions, references, **kwargs)


class JuryBasedEvaluator(Evaluator):
    """
    This is the evaluator component that's based on vanilla Jury scorer.

    Args:
        ```metrics```: ```Union[str, List[str]]```
            What metrics to evaluate?
        ```debug```: ```bool```
            Debug mode flag

    Some of the basic downstream implementation/inheritance using this are:
        - `evaluators.basics.PrecisionEvaluator`
        - `evaluators.basics.RecallEvaluator`
        - `evaluators.basics.F1Evaluator`
        - `evaluators.basics.AccuracyEvaluator`

    Direct usage:

        .. code-block: python

            from evalem.evaluators._base import JuryBasedEvaluator

            references = [
                "Reference 1",
                "Reference 2"
            ]

            predictions = [
                PredictionDTO(text="Reference 1", score=1.0),
                PredictionDTO(text="Reference 2.5", score=0.75)
            ]

            # can use any of the `evaluators.basics.BasicEvaluator`
            evaluator = JuryBasedEvaluator(metrics=["f1", "precision"])
            result = evaluator(predictions=predictions, references=references)
    """

    def __init__(self, metrics: List[str], debug: bool = False) -> None:
        """
        Args:
            ```metrics```: ```Union[str, List[str]]```
                What metrics to evaluate?
            ```debug```: ```bool```
                Debug mode flag
        """
        super().__init__(debug=debug)
        self.scorer = Jury(metrics=metrics)

    def evaluate(
        self,
        predictions: EvaluationPredictionInstance,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> EvaluationOutput:
        predictions = format_to_jury(predictions)
        references = format_to_jury(references)
        return self.scorer(predictions=predictions, references=references)


def main():
    pass


if __name__ == "__main__":
    main()
