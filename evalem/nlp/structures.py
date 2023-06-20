#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .._base.structures import PredictionDTO


@dataclass(frozen=True)
class QuestionAnsweringDTO(PredictionDTO):
    """
    Models the prediction instance for QA.

    For example `models.defaults.DefaultQAModelWrapper` output can be
    an iterable of QAPredictionDTO objects.
    """

    # start index of the answer
    start: Optional[int] = None

    # end index of the answer
    end: Optional[int] = None

    context: Optional[str] = None
    question: Optional[str] = None

    @property
    def text(self) -> str:
        return self.value

    @property
    def answer(self) -> str:
        return self.value
