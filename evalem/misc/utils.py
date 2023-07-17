#!/usr/bin/env python3

from itertools import chain
from typing import Any, Iterable, List, Union

import numpy as np
import pandas as pd
from loguru import logger

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


def flatten_list(nested_list: Union[list, tuple, set]) -> List[Any]:
    """
    Flattens a nested list of list.
    Can take into account in case any element is not a list too
    (eg: `[[1, 2], 3]`)
    """
    return list(
        chain.from_iterable(
            flatten_list(x) if isinstance(x, (list, set, tuple)) else [x]
            for x in nested_list
        ),
    )


def build_comparison_table(
    *eval_pipes,
    inputs,
    references,
    **eval_params,
) -> Union[dict, pd.DataFrame]:
    """
    A utility that runs the provided evaluation pipeline
    and generates a comparison table.

    Note:
        Assumes the same set of inputs and references are run through
        different evaluation pipeline

    Args:
        ```eval_pipes```: ```Type[EvaluationPipeline]```
            Evaluation pipeline objects
        ```inputs```: ```Any```
            Inputs that are fed to each pipeline for forward pass
        ```references```: ```EvaluationReferenceInstance ```
            References/ground-truths for the evaluation.
            See `evalem._base.structures.EvaluationReferenceInstance` for type

    Returns:
        Returns either a pandas DataFrame or dict.
        If pandas dataframe creation fails, it returns a dict.

        For the dataframe, the index is the metric name and other columns
        consist of pipeline name with score value.
    """
    results = map(lambda ep: ep(inputs=inputs, references=references), eval_pipes)
    comparison_map = {}
    dfs = []
    n_items_tracker = []
    for idx, (ep, result) in enumerate(zip(eval_pipes, results)):
        name = f"eval-pipe-{idx}" if not hasattr(ep, "name") else ep.name
        metrics = set(flatten_list(result))
        n_items_tracker.extend([m.total_items for m in metrics])
        comparison_map[name] = metrics

        df = pd.DataFrame(
            map(lambda m: {"metric": m.metric_name, name: m.score}, metrics),
        )
        df.set_index("metric", inplace=True)
        dfs.append(df)
    logger.info(
        f"{int(np.mean(n_items_tracker))} total items are evaluated on average.",
    )
    res = comparison_map
    try:
        res = pd.concat(dfs, join="outer", axis=1)
    except:  # noqa
        logger.warning("Failed to create pd.DataFrame table. Fallback to dict")
    return res
