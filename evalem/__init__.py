__version__ = "0.0.3-alpha.1"

from ._base.evaluators import Evaluator  # noqa
from ._base.pipelines import EvaluationPipeline, SimpleEvaluationPipeline  # noqa


class BaseMetrics:
    """
    We use this to encapsulate base metrics that are task agnostic
    """

    from ._base.metrics import (
        AccuracyMetric,
        ConfusionMatrix,
        F1Metric,
        PrecisionMetric,
        RecallMetric,
    )  # noqa
