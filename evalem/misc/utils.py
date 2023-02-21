#!/usr/bin/env python3

from typing import List, Union

from ..structures import EvaluationDTO, PredictionInstance, ReferenceInstance


def format_to_jury(
    instances: Union[PredictionInstance, ReferenceInstance],
) -> Union[List[str], List[List[str]]]:
    if not instances:
        return []

    # If single DTO object, wrap into a list
    if isinstance(instances, EvaluationDTO):
        return format_to_jury([instances])

    # if single DTO as dict object, wrap a new DTO object into a list
    elif isinstance(instances, dict):
        return format_to_jury([EvaluationDTO.from_dict(instances)])

    elif isinstance(instances, list) and isinstance(instances[0], dict):
        return format_to_jury(
            list(map(lambda x: EvaluationDTO.from_dict(x), instances)),
        )
    elif isinstance(instances, list) and isinstance(instances[0], EvaluationDTO):
        return list(map(lambda x: x.text, instances))
    elif isinstance(instances, list) and isinstance(instances[0], list):
        return list(map(format_to_jury, instances))
    else:
        return instances
