#!/usr/bin/env python3

from typing import Dict

# todo, implement loading these once datasets up on hf spaces
# from datasets import load_dataset


def get_tornados(
    data_type: str = "validation",
    nsamples: int = 1000,
    shuffle: bool = False,
) -> dict:
    """
    This loads squad v2 dataset using HuggingFace datasets module.

    """
    pass


def get_burn_scars(
    data_type: str = "test",
    nsamples: int = 1000,
    shuffle: bool = False,
) -> Dict[str, list]:
    """
    This loads imdb text classification dataset using HuggingFace datasets module.

    """
    pass


def main():
    pass


if __name__ == "__main__":
    main
