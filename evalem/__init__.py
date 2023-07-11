__version__ = "0.0.4-alpha"

from ._base.evaluators import Evaluator  # noqa
from ._base.pipelines import (  # noqa
    EvaluationPipeline,
    NamedSimpleEvaluationPipeline,
    SimpleEvaluationPipeline,
)
from ._base.structures import MetricResult  # noqa
from .nlp.models import (  # noqa
    QuestionAnsweringHFPipelineWrapper,
    TextClassificationHFPipelineWrapper,
)


class BaseMetrics:
    """
    We use this to encapsulate base metrics that are task agnostic
    """

    from ._base.metrics import (  # noqa
        AccuracyMetric,
        ConfusionMatrix,
        F1Metric,
        PrecisionMetric,
        RecallMetric,
    )


class BaseStructures:
    from ._base.structures import (
        EvaluationDTO,
        PredictionDTO,
        PredictionInstance,
        ReferenceDTO,
        ReferenceInstance,
    )
