#!/usr/bin/env python3

from typing import Iterable, List, Union

from ..structures import EvaluationDTO, PredictionInstance, ReferenceInstance


def format_to_jury(
    instances: Union[PredictionInstance, ReferenceInstance],
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
            return EvaluationDTO(text=instance)
        return instance

    instances = (
        list(map(_dtofy, instances)) if isinstance(instances, Iterable) else instances
    )

    # if not List[list] and only List[Type[EvaluationDTO]]
    if isinstance(instances, list) and isinstance(instances[0], EvaluationDTO):
        return list(map(lambda x: x.text, instances))
    # list of list handler
    elif isinstance(instances, list) and isinstance(instances[0], list):
        instances = list(map(_dtofy, instances))
        return list(map(format_to_jury, instances))
    else:
        return instances
