#!/usr/bin/env python3

from ._base import JuryBasedMetric, Metric


class BasicMetric(Metric):
    """
    This represents generic metric implementation
    which is task-agnostic.
    Note:
        This exists only for the sake of type hierarchy.
    """

    pass


class PrecisionMetric(JuryBasedMetric, BasicMetric):
    def __init__(self) -> None:
        super().__init__(metrics="precision")


class RecallMetric(JuryBasedMetric, BasicMetric):
    def __init__(self) -> None:
        super().__init__(metrics="recall")


class F1Metric(JuryBasedMetric, BasicMetric):
    def __init__(self) -> None:
        super().__init__(metrics="f1")


class AccuracyMetric(JuryBasedMetric, BasicMetric):
    def __init__(self) -> None:
        super().__init__(metrics="accuracy")


def main():
    pass


if __name__ == "__main__":
    main()
