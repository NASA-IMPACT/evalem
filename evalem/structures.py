#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass(frozen=True)
class EvaluationDTO:
    """
    Dataclass to hold prediction/reference instance
    """

    text: str
    score: Optional[Union[float, int]] = None


@dataclass(frozen=True)
class PredictionDTO(EvaluationDTO):
    pass


@dataclass(frozen=True)
class ReferenceDTO(EvaluationDTO):
    pass


PredictionInstance = Union[str, PredictionDTO, dict]
ReferenceInstance = Union[str, PredictionDTO]

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
