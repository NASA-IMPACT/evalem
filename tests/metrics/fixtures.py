# flake8: noqa
#!/usr/bin/env python3

from typing import List

import pytest


@pytest.fixture(autouse=True, scope="module")
def references() -> List[str]:
    return ["A", "B", "C", "D", "A"]


@pytest.fixture(autouse=True, scope="module")
def predictions() -> List[str]:
    return ["A", "B", "C", "D", "B"]


def main():
    pass


if __name__ == "__main__":
    main()
