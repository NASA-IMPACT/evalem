#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass(frozen=True)
class EvaluationDTO:
    """
    Dataclass to hold prediction/reference instance
    """

    text: str
    score: float | int | None = None

    @classmethod
    def from_dict(cls, dct: dict) -> EvaluationDTO:
        return cls(text=dct.get("text"), score=dct.get("score"))


@dataclass(frozen=True)
class PredictionDTO(EvaluationDTO):
    pass


@dataclass(frozen=True)
class ReferenceDTO(EvaluationDTO):
    pass


PredictionInstance = Union[str, PredictionDTO, dict]
ReferenceInstance = Union[str, ReferenceDTO]

SinglePredictionInstance = List[PredictionInstance]
MultiplePredictionInstance = List[List[PredictionInstance]]
EvaluationPredictionInstance = Union[
    SinglePredictionInstance,
    MultiplePredictionInstance,
]

SingleReferenceInstance = List[ReferenceInstance]
MultipleReferenceInstance = List[List[ReferenceInstance]]
EvaluationReferenceInstance = Union[SingleReferenceInstance, MultipleReferenceInstance]

EvaluationOutput = Union[int, float, Dict[str, Union[str, int, float]]]
