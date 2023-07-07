#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch


@dataclass(frozen=True)
class EvaluationDTO:
    # could be image, text, anything
    value: Any
    score: Optional[Union[int, float]] = None

    @classmethod
    def from_dict(cls, dct: dict) -> EvaluationDTO:
        return cls(value=dct.get("value"), score=dct.get("score"))

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PredictionDTO(EvaluationDTO):
    pass


@dataclass(frozen=True)
class ClassificationDTO(EvaluationDTO):
    @property
    def label(self) -> Union[str, int]:
        return self.value


@dataclass(frozen=True)
class ReferenceDTO(EvaluationDTO):
    pass


@dataclass(frozen=True)
class MetricResult:
    score: float
    total_items: int
    metric_name: str
    empty_items: int = 0
    extra: Optional[dict] = None

    @classmethod
    def from_dict(cls, dct: dict) -> MetricResult:
        dct = deepcopy(dct)
        return cls(
            score=dct.pop("score", None),
            total_items=dct.pop("total_items", None),
            metric_name=dct.pop("metric_name", None),
            empty_items=dct.pop("empty_items", 0),
            extra=dct,
        )

    def as_dict(self) -> dict:
        return asdict(self)

    def to_dict(self) -> dict:
        return asdict(self)


ImageTensor = Union[np.ndarray, torch.Tensor]

# Represents type instance for any single downstream prediction
PredictionInstance = Union[
    str,
    Type[PredictionDTO],
    dict,
    ImageTensor,
    Type[ClassificationDTO],
]

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
MetricOutput = Union[int, float, Dict[str, Union[str, int, float]], MetricResult]

PathType = Union[str, Path]
