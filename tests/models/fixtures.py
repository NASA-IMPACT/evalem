#!/usr/bin/env python3

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="module")
def squad_v2_data():
    path = Path(__file__).parent.parent.joinpath("data/squad_v2.json")
    data = {}
    with open(path, "r") as f:
        data = json.load(f)
    return data


@pytest.fixture(autouse=True, scope="module")
def imdb_data():
    path = Path(__file__).parent.parent.joinpath("data/imdb.json")
    data = {}
    with open(path, "r") as f:
        data = json.load(f)
    return data


def main():
    pass


if __name__ == "__main__":
    main()
