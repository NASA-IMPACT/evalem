#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Type, Union


@dataclass(frozen=True)
class EvaluationDTO:
    """
    Dataclass to hold prediction/reference instance
    """

    text: str
    score: Optional[int, float] = None

    @classmethod
    def from_dict(cls, dct: dict) -> EvaluationDTO:
        return cls(text=dct.get("text"), score=dct.get("score"))

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PredictionDTO(EvaluationDTO):
    pass


@dataclass(frozen=True)
class QAPredictionDTO(PredictionDTO):
    """
    Models the prediction instance for QA.

    For example `models.defaults.DefaultQAModelWrapper` output can be
    an iterable of QAPredictionDTO objects.
    """

    # start index of the answer
    start: Optional[int] = None

    # end index of the answer
    end: Optional[int] = None


@dataclass(frozen=True)
class ReferenceDTO(EvaluationDTO):
    pass


# Represents type instance for any single downstream prediction
PredictionInstance = Union[str, Type[PredictionDTO], dict]

# Represents type instance for any single downstream reference/ground-truth
ReferenceInstance = Union[str, Type[ReferenceDTO]]

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
MetricOutput = Union[int, float, Dict[str, Union[str, int, float]]]
