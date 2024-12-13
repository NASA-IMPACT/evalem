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
    SequenceType,
)
from ._base import NLPMetric


class AggregationType(Enum):
    MEAN = "mean"
    AVERAGE = "average"
    MAX = "max"


class LLMAsJudgeMetric(NLPMetric):
    """
    Uses a language model to compute metrics by performing a binary
    classification of prediction matching with the reference.
    Uses N tries and compute the aggregate score for each prediction.

    The prompt can be changed using `prompt` attribute.

    Args:
        ```model```: ```str```
            OpenaAI-api compatible model name.
            Could be:
                - open ai models
                - ollama models
        ```api_base```: ```str```
            Base URL for api requests.
            - openai: https://api.openai.com/v1
            - ollama: https://localhost:11434/v1
            If `/v1` is not present, it will be appended
        ```api_key```: ```Optional[str]```
            API key to make request for compleition
        ```n_tries```: ```int```
            Number of times the judgement is done for scoring.
            The final aggregated scores will be based on `LLMAsJudgeMetric.AggregationType`
        ```prompt```: ```Optional[str]```
            Prompt to use for generating the scores.
            If not provided, defaults to `LLMAsJudgeMetric._prompt`
        ```aggregation_type```: ```Optional[AggregationType]```
            Decides how to aggregate scores from the multiple judgement tries.
            Defaults to `AggregationType.MEAN` if not provided.
        ```max_n```: ```Optional[int]```
            If set, the total number of references or predictions per item.
            This is to reduce LLM calls and thus minimizing scoring time.
            Default behaviour is no truncation when set to `None` or less than 1.
            will be truncated.
            - If single reference, multiple predictions, total number of prediction will
            be truncated
            - If multiple reference, single prediction, total number of
            reference will be truncated
        ```debug```:```bool```
            Boolean flag for debug-mode outputs


    Usage:
        .. code-block: python

            from evalem.nlp import LLMAsJudgeMetric

            model = "ollama/llama3.2:3b"
            api_base = "http://localhost:11434/v1"
            model = "gpt-4o-mini"

            api_base = "https://api.openai.com/v1"

            references=["This is title 1", "This has title 2"]
            predictions=[
                ["Title 1", "title 1 absolutely"],
                ["this is title 3, not title 2"]
            ]

            metric = LLMAsJudgeMetric(
                model=MODEL,
                api_base=API_BASE,
                api_key=os.environ.get("OPENAI_API_KEY"),
                # api_key=None,
                n_tries=3,
                prompt=PROMPT,
                debug=True,
            )
            result = metric.compuate(references=references, predictions=predictions)
    """

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
        max_n: Optional[int] = None,
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug)

        model = self.__clean_model(model)
        api_base = self.__clean_url(api_base)
        self.model = outlines.models.openai(
            model,
            base_url=api_base,
            api_key=api_key,
            config=OpenAIConfig(temperature=temperature),
        )
        self.api_base = api_base
        self.n_tries = n_tries or 1
        self.prompt = prompt or LLMAsJudgeMetric._prompt
        self.aggregation_type = aggregation_type or AggregationType.MEAN
        self._sanity_check_prmopt(self.prompt)
        self.max_n = max_n or None
        if self.max_n:
            logger.warning(
                f"Total number of predictions/references per item will be truncated based on `max_n` value.",
            )

    def _sanity_check_prmopt(self, prompt: str) -> bool:
        if "{prediction}" not in prompt or "{reference}" not in prompt:
            raise ValueError(
                "Missing '{prediction} and '{reference}' placeholders in the prmopt.",
            )
        return True

    def __clean_model(self, model: str) -> str:
        if model.startswith("ollama/"):
            model = model.removeprefix("ollama/")
        return model

    def __clean_url(self, url: str) -> str:
        if not url.endswith("/v1"):
            url = urljoin(url, "v1")
        return url

    @staticmethod
    def _flatten_references(
        predictions,
        references,
        max_n: Optional[int] = None,
    ) -> Tuple[EvaluationPredictionInstance, EvaluationReferenceInstance]:
        if max_n is not None and max_n < 1:
            max_n = None
        res = []
        for preds, refs in zip(predictions, references):
            # multiple predictions, single reference
            if isinstance(preds, SequenceType) and isinstance(refs, str):
                res.extend(list(map(lambda p: (p, refs), preds[slice(max_n)])))

            # single prediction, multiple references
            elif isinstance(preds, str) and isinstance(refs, SequenceType):
                res.extend(list(map(lambda r: (preds, r), refs[slice(max_n)])))

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
        # make sure to flatten
        predictions, references = self._flatten_references(
            predictions,
            references,
            max_n=self.max_n,
        )
        if self.debug:
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
