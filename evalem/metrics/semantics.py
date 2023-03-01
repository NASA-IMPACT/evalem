#!/usr/bin/env python3

from typing import Optional

import numpy as np
from jury.metrics import Bartscore

from ..misc.utils import format_to_jury
from ..structures import (
    EvaluationPredictionInstance,
    EvaluationReferenceInstance,
    MetricOutput,
)
from ._base import JuryBasedMetric


class SemanticMetric(JuryBasedMetric):
    """
    Metric respresenting semantics score between predictions and references.
    """

    pass


class BertScore(SemanticMetric):
    """
    Uses a BERT model to compute the semantic similarity using the contextual
    embeddings from the model.

    Original paper at: https://arxiv.org/abs/1904.09675

    Implementation at: https://github.com/Tiiiger/bert_score

    Args:
        ```model_type```: ```str```
            Which model to use? Check this URL for complete list:
            https://github.com/Tiiiger/bert_score/blob/master/bert_score/utils.py
        ```device```: ```str```
            Which device to run the model on? Defaults to "cpu".
        ```per_instance_score```: ```bool```
            If enabled, precision, recall and f1 score per instance is also
            returned in the computation result.
        ```debug```: ```bool```
            Enable debugging log? Defaults to False.

    Usage:
        .. code-block: python

            from evalem.metrics import BertScore

            references = [
                "Reference 1",
                "Reference 2"
            ]

            predictions = [
                PredictionDTO(text="Reference 1", score=1.0),
                PredictionDTO(text="Reference 2.5", score=0.75)
            ]

            # default
            scorer = BertScore()

            # with another model on GPU
            scorer = BertScore(
                model_type="distilbert-base-uncased",
                device="cuda:0"
            )
            result = scorer(predictions=predictions, references=references)
    """

    def __init__(
        self,
        model_type: str = "roberta-large",
        device: str = "cpu",
        per_instance_score: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(metrics="bertscore", device=device, debug=debug)
        self.model_type = model_type
        self.per_instance_score = per_instance_score

    def compute(
        self,
        predictions: EvaluationPredictionInstance,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> MetricOutput:
        device = kwargs.pop("device", self.device)
        model_type = kwargs.pop("model_type", self.model_type)
        result = super().compute(
            predictions=predictions,
            references=references,
            model_type=model_type,
            device=device,
            **kwargs,
        )
        if not self.per_instance_score:
            result["bertscore"].pop("precision", None)
            result["bertscore"].pop("recall", None)
            result["bertscore"].pop("f1", None)
        return result


class BartScore(SemanticMetric):
    """
    This uses BART model (an encoder-decoder model) to compute the
    semantic similarity.

    Paper: https://arxiv.org/pdf/2106.11520.pdf

    Args:
        ```model_checkpoint```: ```str```
            Which checkpoint to use? Defaults to `bartscore-large-cnn`
        ```model_weights```: ```Optional[str]```
            Optional model weights.
            If provided, will override themodel_checkpoint
        ```device```: ```str```
            Which device to run the model on? Defaults to "cpu".
        ```max_length```: ```int```
            Max number of tokens in a text. Defaults to 1024.
        ```debug```: ```bool```
            Enable debugging log? Defaults to False.
    """

    def __init__(
        self,
        model_checkpoint: str = "bartscore-large-cnn",
        model_weights: Optional[str] = None,
        device: str = "cpu",
        max_length: int = 1024,
        debug: bool = False,
    ) -> None:
        super().__init__(metrics=None, device=device, debug=debug)
        # we use this method to have more flexibility in creating the object
        self.scorer = Bartscore.construct(
            device=device,
            model_checkpoint=model_checkpoint,
            model_weights=model_weights,
            max_length=max_length,
        )

    def compute(
        self,
        predictions: EvaluationPredictionInstance,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> MetricOutput:
        predictions = format_to_jury(predictions)
        references = format_to_jury(references)

        predictions, references = self._flatten_references(predictions, references)

        # Low-level access to Bartscorer directly
        # See: https://github.com/neulab/BARTScore
        score = np.mean(self.scorer.scorer.score(predictions, references, **kwargs))
        return {
            "bartscore": dict(
                score=score,
                model_checkpoint=self.scorer.model_checkpoint,
                model_weights=self.scorer.model_weights,
                total_items=len(predictions),
                flattened=True,
            ),
        }


def main():
    pass


if __name__ == "__main__":
    main()
