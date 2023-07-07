#!/usr/bin/env python3

from itertools import count
from typing import Iterable, List, Union

from .._base.structures import EvaluationDTO, PredictionInstance, ReferenceInstance


def format_to_jury(
    instances: Union[PredictionInstance, ReferenceInstance],
    stringify: bool = True,
) -> Union[List[str], List[List[str]]]:
    if not instances:
        return []

    if isinstance(instances, (EvaluationDTO, dict, str)):
        return format_to_jury([instances])

    # converts everything to DTO
    def _dtofy(instance):
        if isinstance(instance, dict):
            return EvaluationDTO.from_dict(instance)
        if isinstance(instance, str):
            return EvaluationDTO(value=instance)
        if stringify and isinstance(instance, int):
            return EvaluationDTO(value=str(instance))
        return instance

    instances = (
        list(map(_dtofy, instances)) if isinstance(instances, Iterable) else instances
    )

    # if not List[list] and only List[Type[EvaluationDTO]]
    if isinstance(instances, list) and isinstance(instances[0], EvaluationDTO):
        return list(map(lambda x: x.value, instances))
    # list of list handler
    elif isinstance(instances, list) and isinstance(instances[0], list):
        instances = list(map(_dtofy, instances))
        return list(map(format_to_jury, instances))
    else:
        return instances


class InstanceCountMixin:
    _ids = count(0)
    _names = set()

    def __init__(self):
        self.idx = next(self._ids)
        self._name = None

    @property
    def name(self):
        return self._name or f"{self.__class__.__name__}:{self.idx}"

    @name.setter
    def name(self, name: str):
        self._name = name
