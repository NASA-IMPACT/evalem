#!/usr/bin/env python3

from abc import abstractmethod
from typing import Any

from .._base._base import AbstractBase


class Pipeline(AbstractBase):
    """
    Represents a type for Pipeline component.
    All the downstream pipeline object should implement the `.run(...)` method.

    See `pipelines.defaults.SimpleEvaluationPipeline` for an implementation.
    """

    @abstractmethod
    def run(self, *args, **kwags) -> Any:
        """
        Entry-point method to run the evaluation.
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> Any:
        return self.run(*args, **kwargs)


def main():
    pass


if __name__ == "__main__":
    main()
