#!/usr/bin/env python3

from typing import List, Union

from ..structures import EvaluationDTO, PredictionInstance, ReferenceInstance


def format_to_jury(
    instances: Union[PredictionInstance, ReferenceInstance],
) -> Union[List[str], List[List[str]]]:
    if not instances:
        return []

    if isinstance(instances, EvaluationDTO):
        instances = [instances]
    elif isinstance(instances, dict):
        instances = [EvaluationDTO.from_dict(instances)]
    elif isinstance(instances, list) and isinstance(instances[0], EvaluationDTO):
        return list(map(lambda x: x.text, instances))
    elif isinstance(instances, list) and isinstance(instances[0], list):
        return list(map(format_to_jury, instances))
    else:
        return instances
