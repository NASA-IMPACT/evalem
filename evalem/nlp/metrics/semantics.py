#!/usr/bin/env python3


import dataclasses
from typing import Optional

import numpy as np
from jury.metrics import Bartscore

from ..._base.metrics import JuryBasedMetric
from ..._base.structures import (
    EvaluationPredictionInstance,
    EvaluationReferenceInstance,
    MetricResult,
)
from ...misc.utils import format_to_jury
from ._base import NLPMetric


class SemanticMetric(NLPMetric):
    """
    Metric respresenting semantics score between predictions and references.
    """

    pass


class BertScore(JuryBasedMetric, SemanticMetric):
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
            Else: mean precision, recall and f1 is computed by default.
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
        model_type: str = "bert-base-uncased",
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
    ) -> MetricResult:
        device = kwargs.pop("device", self.device)
        model_type = kwargs.pop("model_type", self.model_type)
        result = super().compute(
            predictions=predictions,
            references=references,
            model_type=model_type,
            device=device,
            **kwargs,
        )
        # if you want to supress a list of all these metrics
        # and want to just have mean/average.
        if not self.per_instance_score:
            for _key in ["precision", "recall", "f1"]:
                result.extra["bertscore"][_key] = np.mean(
                    result.extra["bertscore"][_key],
                )
        return result


class BartScore(JuryBasedMetric, SemanticMetric):
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
    ) -> MetricResult:
        predictions = format_to_jury(predictions)
        references = format_to_jury(references)

        predictions, references = self._flatten_references(predictions, references)

        # Low-level access to Bartscorer directly
        # See: https://github.com/neulab/BARTScore
        score = np.mean(self.scorer.scorer.score(predictions, references, **kwargs))
        return MetricResult(
            score=score,
            total_items=len(predictions),
            metric_name="BartScore",
            extra=dict(
                flattened=True,
                model_checkpoint=self.scorer.model_checkpoint,
            ),
        )


class BleuMetric(JuryBasedMetric, SemanticMetric):
    """
    Bilingual Evaluation Understudy (BLEU) is generally used for
    text-translation.

    References:
        - https://en.wikipedia.org/wiki/BLEU
        - https://aclanthology.org/P02-1040.pdf

    Usage:

        .. code-block: python

            from evalem.metrics import BleuMetric

            metric = BleuMetric()
            results = metric(predictions=predictions, references=references)
    """

    def __init__(self) -> None:
        super().__init__(metrics="bleu")


class SacreBleuMetric(JuryBasedMetric, SemanticMetric):
    def __init__(self) -> None:
        super().__init__(metrics="sacrebleu")


class MeteorMetric(JuryBasedMetric, SemanticMetric):
    """
    Metric for Evaluation of Translation with Explicit ORdering.

    References:
        - https://en.wikipedia.org/wiki/METEOR
        - https://www.cs.cmu.edu/~alavie/METEOR/pdf/Banerjee-Lavie-2005-METEOR.pdf
        - https://arxiv.org/abs/2109.14250

    Usage:

        .. code-block: python

            from evalem.metrics import MeteorMetric

            metric = MeteorMetric()
            results = metric(predictions=predictions, references=references)
    """

    def __init__(self) -> None:
        super().__init__(metrics="meteor")


class RougeMetric(JuryBasedMetric, SemanticMetric):
    """
    Recall-Oriented Understudy for Gisting Evaluation.

    References:
        - https://en.wikipedia.org/wiki/ROUGE_(metric)
        - https://aclanthology.org/W04-1013.pdf
        - https://arxiv.org/abs/2109.14250

    Usage:

        .. code-block: python

            from evalem.metrics import RougeMetric

            metric = RougeMetric()
            results = metric(predictions=predictions, references=references)

    """

    def __init__(self) -> None:
        super().__init__(metrics="rouge")

    def compute(
        self,
        predictions: EvaluationPredictionInstance,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> MetricResult:
        result = super().compute(
            predictions=predictions,
            references=references,
            **kwargs,
        )
        score = float(np.mean(list((result.extra or {}).get("rouge", {}).values())))
        return dataclasses.replace(result, score=score)


def main():
    pass


if __name__ == "__main__":
    main()
