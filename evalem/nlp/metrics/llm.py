#!/usr/bin/env python3

from enum import Enum
from typing import List, Optional, Tuple
from urllib.parse import urljoin

import numpy as np
import outlines
from loguru import logger
from outlines.models.openai import OpenAIConfig

from ..._base.structures import (
    EvaluationPredictionInstance,
    EvaluationReferenceInstance,
    MetricResult,
)
from ._base import NLPMetric


class AggregationType(Enum):
    MEAN = "mean"
    AVERAGE = "average"
    MAX = "max"


class LLMAsJudgeMetric(NLPMetric):
    _prompt = (
        "You are a very good binary classifier."
        + " Classify the quality of prediction based on the provided reference.\n"
        + "Prediction: {prediction}\n"
        + "Reference: {reference}"
    )

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: Optional[str] = None,
        n_tries: int = 1,
        temperature: float = 0.0,
        prompt: Optional[str] = None,
        aggregation_type: Optional[List[AggregationType]] = None,
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug)
        self.model = outlines.models.openai(
            self.__clean_model(model),
            base_url=api_base,
            api_key=api_key,
            config=OpenAIConfig(temperature=temperature),
        )
        self.api_base = self.__clean_url(api_base)
        self.n_tries = n_tries or 1
        self.prompt = prompt or LLMAsJudgeMetric._prompt
        self.aggregation_type = aggregation_type or AggregationType.MEAN

    def __clean_model(self, model: str) -> str:
        if model.startswith("ollama/"):
            model = model.removeprefix("ollama/")
        return model

    def __clean_url(self, url: str) -> str:
        if not url.endswith("/v1"):
            url = urljoin(url, "/v1")
        return url

    @staticmethod
    def _flatten_references(
        predictions,
        references,
    ) -> Tuple[EvaluationPredictionInstance, EvaluationReferenceInstance]:
        res = []
        for preds, refs in zip(predictions, references):
            # multiple predictions, single reference
            if isinstance(preds, (list, tuple, set)) and isinstance(refs, str):
                res.extend(list(map(lambda p: (p, refs), preds)))

            # single prediction, multiple references
            elif isinstance(preds, str) and isinstance(refs, (list, tuple, set)):
                res.extend(list(map(lambda r: (preds, r), refs)))

            # single prediction, single reference
            else:
                res.append((preds, refs))

        predictions, references = zip(*res)
        return predictions, references

    def compute(
        self,
        predictions: EvaluationPredictionInstance,
        references: EvaluationReferenceInstance,
        **kwargs,
    ) -> MetricResult:
        predictions, references = self._flatten_references(predictions, references)
        logger.debug(f"Evaluating for {len(predictions)} predictions.")
        generator = outlines.generate.choice(self.model, ["0", "1"])
        res = []
        individual_scores = []
        for pred, ref in zip(predictions, references):
            prompt = self.prompt.format(prediction=pred, reference=ref)
            if self.debug:
                logger.debug(f"Prompt :: {prompt}")
            scores = []
            score = np.nan
            with outlines.caching.cache_disabled():
                scores = self._compute_single(generator, prompt, self.n_tries)
            score = self._aggregate_scores(scores, self.aggregation_type)
            individual_scores.append(scores)
            res.append(score)
            if self.debug:
                logger.debug(f"Scores :: {scores}")
                logger.debug(f"Aggregated score :: {score}")
        return MetricResult(
            score=float(np.mean(res)),
            total_items=len(predictions),
            metric_name=self.__classname__,
            extra=dict(scores=individual_scores, model=self.model),
        )

    @staticmethod
    def _aggregate_scores(
        scores: List[int],
        aggregation_type: AggregationType = AggregationType.MEAN,
    ) -> float:
        if not scores:
            return 0.0
        res = 0.0
        if aggregation_type in [AggregationType.MEAN, AggregationType.AVERAGE]:
            res = round(sum(scores) / len(scores), 4)
        elif aggregation_type in [AggregationType.MAX]:
            res = float(max(scores))
        return res

    def _compute_single(self, generator, prompt, n_tries) -> List[float]:
        return [int(generator(prompt)) for n in range(n_tries)]


def main():
    pass


if __name__ == "__main__":
    main()
