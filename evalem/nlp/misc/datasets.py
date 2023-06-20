#!/usr/bin/env python3

from typing import Dict

from datasets import load_dataset


def get_squad_v2(
    data_type: str = "validation",
    nsamples: int = 1000,
    shuffle: bool = False,
) -> dict:
    """
    This loads squad v2 dataset using HuggingFace datasets module.

    Args:
        ```data_type```: ```str```
            Either "train" or "validation"
        ```nsamples```: ```int```
            How many samples to load?
            Note: The returned data size may not be exactly equal to nsamples
            as we're filtering out empty references
        ```shuffle```: ```bool```
            If enabled, shuffles the data prior to sampling/filtering.

    Returns:
        Returns a dict with 2 keys:
            - `inputs`: `List[dict]`, each dict has "context" and "question"
            keys
            - `references`: ```List[List[str]]```

    """
    nsamples = nsamples or 0

    data = load_dataset("squad_v2")[data_type]
    data = data.shuffle(seed=42) if shuffle else data
    data = data.select(range(nsamples)) if nsamples > 0 else data

    inputs = [dict(question=d["question"].lstrip(), context=d["context"]) for d in data]
    references = [d["answers"]["text"] for d in data]

    inputs, references = zip(*filter(lambda x: len(x[1]) > 0, zip(inputs, references)))
    return dict(inputs=inputs, references=references)


def get_imdb(
    data_type: str = "test",
    nsamples: int = 1000,
    shuffle: bool = False,
) -> Dict[str, list]:
    """
    This loads imdb text classification dataset using HuggingFace datasets module.

    Args:
        ```data_type```: ```str```
            Either "train" or "test"
        ```nsamples```: ```int```
            How many samples to load?
            Note: The returned data size may not be exactly equal to nsamples
            as we're filtering out empty references
        ```shuffle```: ```bool```
            If enabled, shuffles the data prior to sampling/filtering.

    Returns:
        Returns a dict with 2 keys:
            - `inputs`: `List[dict]`, each dict has "context" and "question"
            keys
            - `references`: ```List[List[str]]```

    """
    nsamples = nsamples or 0
    data = load_dataset("imdb")[data_type]
    data = data.shuffle(seed=42) if shuffle else data
    data = data.select(range(nsamples)) if nsamples > 0 else data

    label_map = ["NEGATIVE", "POSITIVE"]
    inputs = [(d["text"], label_map[d["label"]]) for d in data]
    inputs, references = zip(*inputs)
    return dict(inputs=list(inputs), references=list(references))


def main():
    pass


if __name__ == "__main__":
    main
