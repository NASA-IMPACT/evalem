#!/usr/bin/env python3

from abc import abstractmethod

from .._base import AbstractBase
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


def main():
    pass


if __name__ == "__main__":
    main()
