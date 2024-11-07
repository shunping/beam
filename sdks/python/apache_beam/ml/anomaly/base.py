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
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
from typing import Generic
from typing import Iterable
from typing import List
from typing import Optional
from typing import TypeVar

EPSILON = 1e-12

ExampleT = TypeVar('ExampleT')
ScoreT = TypeVar('ScoreT')
LabelT = TypeVar('LabelT')
AggregateT = TypeVar('AggregateT')


@dataclass(frozen=True)
class AnomalyPrediction(Generic[ScoreT, LabelT]):
  model_id: Optional[str] = ""
  score: Optional[ScoreT] = None
  label: Optional[LabelT] = None
  threshold: Optional[ScoreT] = None
  info: str = ""
  agg_history: Optional[List[AnomalyPrediction]] = None


@dataclass(frozen=True)
class AnomalyResult(Generic[ExampleT, ScoreT, LabelT]):
  example: ExampleT
  prediction: AnomalyPrediction[ScoreT, LabelT]


class AnomalyModel(ABC, Generic[ExampleT, ScoreT]):

  @abstractmethod
  def learn_one(self, x: ExampleT) -> None:
    raise NotImplementedError

  @abstractmethod
  def score_one(self, x: ExampleT) -> ScoreT:
    raise NotImplementedError


class ThresholdFunc(Generic[ScoreT, LabelT]):

  def __init__(self,
               normal_label=None,
               outlier_label=None,
               label_class: type[LabelT] = int):
    self._normal_label = label_class(0) if normal_label is None else normal_label  # type: ignore
    self._outlier_label = label_class(1) if outlier_label is None else outlier_label  # type: ignore
    self._threshold = None

  def __call__(self, score: ScoreT) -> LabelT:
    raise NotImplementedError

  @property
  def is_stateful(self) -> bool:
    raise NotImplementedError

  @property
  def threshold(self) -> Optional[ScoreT]:
    return self._threshold


class AggregationFunc(Generic[AggregateT]):

  def __init__(self,
               agg_func: Callable[[Iterable[AggregateT]], AggregateT],
               include_history=False,
               model_override=""):
    self._agg_func = agg_func
    self._include_history = include_history
    self._model_override = model_override

  def __call__(self,
               predictions: Iterable[AnomalyPrediction]) -> AnomalyPrediction:
    raise NotImplementedError
