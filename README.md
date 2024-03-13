# evalem
Machine Learning pipeline evaluation framework


# Contributing

We use [Hatch](https://hatch.pypa.io/latest/install/) for environment management and packaging `evalem`.

To start the development env, you'll first need to install hatch. We recommend installing with [pipx](https://github.com/pypa/pipx) so it doesn't interfere with other python envs.

```
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```

This globally installs hatch, restart the terminal for changes to take effect. Then,

`pipx install hatch`

With hatch installed, you can start a shell environment with test dependencies, creating the env if it doesn't already exist.

`hatch -e test shell`

See the pyproject.toml for the test dependencies. The test environment inherits dependencies from the default environment.

If you need to remove dependencies, you'll need to delete the environments and recreate them. Use

`hatch env prune` to remove all environments and `hatch -e test shell' to recreate the default and test environments.


# Installing locally

```
git@github.com:NASA-IMPACT/evalem.git

pip install -e .
```

# DTOs

Base DTOs exist at `evalem._base.structures`, primarily which consists of `PredictionDTO` and `ReferenceDTO`.

Different evaluation mode can happen during metric calculation:

- Single Reference, Single Prediction  (SRSP)
- Single Reference, Multiple Predictions (SRMP)
- Multiple References, Single Prediction (MRSP)
- Multiple References, Multiple Prediction (MRMP)

out of which, SRSP and MRSP seems like common mode of evaluation.

Out-of-box, evalem metrics bake in all these and they transform these references/predictions internally based on the metric. For multiple references, common mode is to flatten everything to single list, duplicating the prediction value to different ground truths.



# Metrics

Evalem provides a handful of evaluation metrics out-of-box, in a single place so that users won't have to switch between different implementation, between different external modules and packages.

All the metrics are based off the main base `evalem._base.metrics.Metric`.

All the metrics object take in `references` and `predictions`during execution/runtime/evaluation-time. Each metric could have different way of building. For instance, some metrics could be run on different devices other than `cpu`(such as `cuda: 0`, `mps`, etc based on their implementation). Each metric evaluation gives result in a `evalem._base.structures.MetricResult` structure which consist of `score` and other parameters.

## Base Metrics

`evalem._base.metrics` module has some base metrics (of type `BasicMetric`) such as:

- `PrecisionMetric`
- `RecallMetric`
- `F1Metric`
- `AccuracyMetric`
- `ConfusionMatrix`

```python
from evalem._base.structures import (
    PredictionDTO,
    ReferenceDTO,
)
from evalem._base.structures import MetricResult

from evalem._base.metrics import (
    Metric,
    AccuracyMetric,
    PrecisionMetric,
    RecallMetric,
    F1Metric,
    BasicMetric,
    ConfusionMatrix
)

# Single References (SR)
references = [
    ReferenceDTO(value="Reference 1"),
    ReferenceDTO(value="Reference 2")
]

# Multiple references
references = [
    [
        ReferenceDTO("Reference 1"),
        dict(value="Reference 1.1"),
                dict(value="Reference 2"),

    ],
    [
        dict(value="Reference 2"),
        ReferenceDTO("Dummy 1"),
        ReferenceDTO("Dummy 2"),
        "Dummy 3"
    ]
]

predictions = [
    PredictionDTO(value="Reference 1", score=1.0),
    PredictionDTO(value="Reference 2.5", score=0.75)
]

print(PrecisionMetric()(references=references, predictions=predictions)) # Output of type MetricResult
print(RecallMetric()(references=references, predictions=predictions))
print(AccuracyMetric()(references=references, predictions=predictions))
print(ConfusionMatrix()(references=references, predictions=predictions))
```





## NLP metrics

`evalem.nlp.metrics` separates out the namespace for NLP.

All the NLP-based metrics are derived from `evalem.nlp.metrics._base.NLPMetric` which itself inherits from the base `Metric`.

Some of the available metrics are:

- `evalem.nlp.metrics.BartScore`
- `evalem.nlp.metrics.BertScore`
- `evalem.nlp.metrics.BleuMetric`
- `evalem.nlp.metrics.ExactMatchMetric`
- `evalem.nlp.metrics.MeteorMetric`
- `evalem.nlp.metrics.RougeMetric`
- `evalem.nlp.metrics.SacredBleuMetric`



All the semantic metrics are of type `evalem.nlp.metrics.semantics.SemanticMetric` which is obviously inherited from `NLPMetric`



```python
from evalem.nlp.metrics import NLPMetric, SemanticMetric

from evalem.nlp.metrics import (
    BartScore,
    BertScore,
    BleuMetric,
    ExactMatchMetric,
    MeteorMetric,
    RougeMetric,
    SacreBleuMetric,

)

references = [
    "I love NLP",
    "I love working with language models",
    "I love my cat"
]

predictions = [
    "I don't really like doing NLP",
    "Language models are okay",
    "I absolutely love my cat"
]

print(BertScore(device="cpu")(references=references, predictions=predictions))
print(ExactMatchMetric()(references=references, predictions=predictions))
...
```



# Evaluators

Evaluators in evalem help in containerizing metrics to run them in single go instead of having to create separate instances for each metric. It's one level of abstraction above the metric.

The base evaluator container can be : `evalem._base.evaluators.Evaluator` which takes in a list of metric objects to run in single go.

We can compose any metrics together and run them in single go for a set of datasets (references, predictions)



```python
from evalem._base.evaluators import Evaluator

evaluator = Evaluator(metrics=[
    BertScore(device="cpu"),
    BartScore(device="cpu"),
    ExactMatchMetric(),
    MeteorMetric(),
    RougeMetric(),
    SacreBleuMetric(),
    BleuMetric()
])

result = evaluator_nlp(references=references, predictions=predictions) # Outputs a list of MetricResult objects
print(result)
```



# Model Wrappers

evalem also provivdes a way to evaluate models directly by runnin inputs through the models, getting predictions and evaluating based on the references. To standardize the model forward-pass, evalem has model wrappers `evalem._base.models.ModelWrapper`. All model wrappers take in an arbitrary model and the forward pass has to be implemented by downstream wrapper implementation (by implementing `_predict(...) method`).

`evalem._base.models.HFWrapper` provides a type for Huggingface-based models



## Model wrappers implementations

- `evalem.nlp.models.HFLMWrapper`: wrapper for upstream Huggingface language model
- `evalem.nlp.models.HFPipelineWrapper`: wrapper for huggingface pipeline (which itself wraps model + tokenizer)



`evalem.nlp.models.QuestionAnsweringHFPipelineWrapper`and `evalem.nlp.models.TextClassificationHFPipelineWrapper`are two impelmentations for QA and Text classification within evalem eco-system.

# QA



## QA Model Wrapper

`QuestionAnsweringHFPipelineWrapper` provides a wrapper for QA tasks.

```python
from evalem.nlp.models import QuestionAnsweringHFPipelineWrapper
from evalem.nlp.structures import QuestionAnsweringDTO

model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")

wrapped_model = QuestionAnsweringHFPipelineWrapper(
    model=model,
    tokenizer=tokenizer,
    device="mps"
)

data = [
    {
        "context": "Deep within the labyrinthine caves, echoes tell tales of lost civilizations.",
        "question": "What sails toward the unknown?",
        "reference": "A lone ship"
    },
    {
        "context": "Beneath the bustling city streets, forgotten catacombs hold secrets of the past.",
        "question": "What watches the skies?",
        "reference": "The old observatory"
    },
    {
        "context": "In the heart of the ancient forest, a hidden lake glimmers under the moonlight.",
        "question": "What lies in the heart of the forest?",
        "reference": "A hidden lake"
    },
    {
        "context": "Beneath the bustling city streets, forgotten catacombs hold secrets of the past.",
        "question": "What lies in the heart of the forest?",
        "reference": "Forgotten catacombs"
    },
    {
        "context": "Deep within the labyrinthine caves, echoes tell tales of lost civilizations.",
        "question": "What lies in the heart of the forest?",
        "reference": "Labyrinthine caves"
    }
]

pprint(wrapped_model([ dict(context=d["context"], question=d["context"]) for d in data]))


```



## QA Evaluation

We can directly utilize all the previous evaluation metric, do forward-pass manually and provide predictions, references manually to compute the metric.

To abstract the forward pass itself, evalem provides `SimpleEvaluationPipeline` that takes in a wrapped model and list of evaluators and run them directly.

```python
from evalem import SimpleEvaluationPipeline
from evalem.nlp.evaluators import QAEvaluator

# Out-of-box QA evaluation, consists of basic metrics
evaluator = QAEvaluator()
result = evaluator(
	references=references,
  predictions=wrapped_model(inputs)
)

# wrap into a pipeline
eval_pipe = SimpleEvaluationPipeline(
    model=wrapped_model,
    evaluators=[QAEvaluator()]
)
result = eval_pipe(
	inputs,
  references,
  ...
)
```



# Text Classification



## Text Classification Wrapper

`TextClassificationHFPipelineWrapper` provides an abstraction for text classification models.

```python
from transformers import AutoModelForSequenceClassification
from evalem.nlp.models import TextClassificationHFPipelineWrapper

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

wrapped_model = TextClassificationHFPipelineWrapper(
    model=model,
    tokenizer=tokenizer
)

pprint(wrapped_model("I don't like mangoes"))
```



## Text Classification Evaluation

```python
from evalem import SimpleEvaluationPipeline
from evalem.nlp.evaluators import TextClassificationEvaluator

inputs = [
    "I love NLP",
    "I love working with language models",
    "I love my cat",
    "I don't like mangoes"
]

references = [
    "POSITIVE",
    "POSITIVE",
    "POSITIVE",
    "NEGATIVE"
]

# Out-of-box QA evaluation, consists of basic metrics
evaluator = TextClassificationEvaluator()

result = evaluator(
	references=references,
  predictions=wrapped_model(inputs)
)

# wrap into a pipeline
eval_pipe = SimpleEvaluationPipeline(
    model=wrapped_model,
    evaluators=[TextClassificationEvaluator()]
)
result = eval_pipe(
	inputs,
  references,
  ...
)
```



# Notes

Everywhere that takes in `Evaluator` type, we can basically compose any evaluators and run them through the `EvaluationPipeline`.

For isinstance:

```python
from evalem._base.evaluators import Evaluator
from evalem import SimpleEvaluationPipeline

# build model wrapper
wrapped_model = ...

# get dataset
inputs = ...
references = ...

evaluators = [
    Evaluator(metrics=[
        AccuracyMetric(),
        ConfusionMatrix(),
        ExactMatchMetric(),
        F1Metric(),
    ]),
    Evaluator(metrics=[
        BertScore(),
        BartScore()
    ])
]

eval_pipe = SimpleEvaluationPipeline(
    model = wrapped_model,
    evaluators=evaluators
)

result = eval_pipe(
	inputs,
  references,
  ...
)

pprint(result)
```
