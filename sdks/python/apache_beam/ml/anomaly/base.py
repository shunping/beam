#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Base classes for anomaly detection"""

from abc import ABC, abstractmethod
import dataclasses
from dataclasses import dataclass
import itertools
import math
from typing import Callable
from typing import Optional
from typing import Generic
from typing import Protocol
from typing import Iterable
from typing import TypeVar
from typing import Union

import apache_beam as beam

EPSILON = 1e-12

ExampleT = TypeVar('ExampleT')
ScoreT = TypeVar('ScoreT')
LabelT = TypeVar('LabelT')

@dataclass(frozen=True)
class AnomalyPrediction(Generic[ScoreT, LabelT]):
  model_id: Optional[str] = ""
  score: Optional[ScoreT] = None
  label: Optional[LabelT] = None
  threshold: Optional[ScoreT] = None
  auc: Optional[float] = None
  info: str = ''


@dataclass(frozen=True)
class AnomalyResult(Generic[ExampleT, ScoreT, LabelT]):
  example: ExampleT
  prediction: AnomalyPrediction[ScoreT, LabelT]


class AnomalyModel(ABC, Generic[ExampleT, ScoreT]):
  @abstractmethod
  def learn_one(self, x: ExampleT) -> None:
    ...

  @abstractmethod
  def score_one(self, x: ExampleT) -> ScoreT:
    ...


class BaseThresholdFunc(beam.DoFn):
  @property
  def threshold(self) -> Union[int, float]:
    raise NotImplementedError

  def _update_prediction(
      self, result: AnomalyResult) -> AnomalyResult:
    if result.prediction.score is None:
      label = 0
    else:
      label: int = 0 if result.prediction.score < self.threshold else 1
    return dataclasses.replace(
        result,
        prediction=dataclasses.replace(
            result.prediction, label=label, threshold=self.threshold))


class BaseAggregation(Protocol):
  def __call__(self, predictions: Iterable[AnomalyPrediction]) -> AnomalyPrediction:
    ...

class ScoreAggregation(BaseAggregation):

  def __init__(self,
               aggregate_func: Callable[[Iterable[float]], float],
               include_history=False,
               model_override=""):
    self.aggregate_scores = aggregate_func
    self._include_history = include_history
    self._model_override = model_override

  # def aggregate_scores(self, predictions: Iterable[float]):
  #   raise NotImplementedError

  def __call__(self, decisions: Iterable[AnomalyPrediction[float, LabelT]]) -> AnomalyPrediction[float, LabelT]:
    scores = list(itertools.filterfalse(
        lambda score: score is None or math.isnan(score),
        map(lambda decision: decision.score, decisions)))

    if len(scores) == 0:
      return AnomalyPrediction(model_id=self._model_override)

    score = self.aggregate_scores(scores)  # type: ignore

    info = ('[' + ('; '.join(map(str, decisions))) +
            ']') if self._include_history else ''

    return AnomalyPrediction(
        model_id=self._model_override, score=score, info=info)