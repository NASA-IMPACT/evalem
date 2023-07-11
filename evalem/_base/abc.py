#!/usr/bin/env python3

from abc import ABC
from itertools import count
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


class InstanceCountMixin:
    """
    This mixin is used to autogenerate names for
    individual object.
    """

    _ids = count(0)

    def __init__(self):
        self.idx = next(self._ids)
        self._name = None

    @property
    def name(self):
        return self._name or f"{self.__class__.__name__}:{self.idx}"

    @name.setter
    def name(self, name: str):
        self._name = name


def main():
    pass


if __name__ == "__main__":
    main()
