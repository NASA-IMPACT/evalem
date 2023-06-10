#!/usr/bin/env python3

from abc import ABC
from typing import Any


class AbstractBase(ABC):
    """
    An Abstract Base Class with some barebone methods.
    """

    def __init__(self, debug: bool = False) -> None:
        self.debug = bool(debug)

    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    @property
    def __classname__(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:
        return f"[{self.__classname__}]"

    def __repr__(self) -> str:
        return f"[{self.__classname__}]"


def main():
    pass


if __name__ == "__main__":
    main()
