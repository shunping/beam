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

import abc
from dataclasses import dataclass
from typing import Generic
from typing import Iterable
from typing import Optional
from typing import TypeVar

ExampleT = TypeVar('ExampleT')
ScoreT = TypeVar('ScoreT')
LabelT = TypeVar('LabelT')


@dataclass(frozen=True)
class AnomalyPrediction(Generic[ScoreT, LabelT]):
  model_id: Optional[str] = None
  score: Optional[ScoreT] = None
  label: Optional[LabelT] = None
  threshold: Optional[ScoreT] = None
  info: str = ""
  agg_history: Optional[Iterable[AnomalyPrediction[ScoreT, LabelT]]] = None


@dataclass(frozen=True)
class AnomalyResult(Generic[ExampleT, ScoreT, LabelT]):
  example: ExampleT
  prediction: AnomalyPrediction[ScoreT, LabelT]


class AnomalyModel(abc.ABC, Generic[ExampleT, ScoreT]):
  def __init__(
      self,
      features: Optional[Iterable[str]] = None,
      target: Optional[str] = None):
    self._features = features
    self._target = target

  # assume example type is the same as model input type
  @abc.abstractmethod
  def get_x(self, data: ExampleT) -> ExampleT:
    raise NotImplementedError

  @abc.abstractmethod
  def learn_one(self, x: ExampleT) -> None:
    raise NotImplementedError

  @abc.abstractmethod
  def score_one(self, x: ExampleT) -> ScoreT:
    raise NotImplementedError


class ThresholdFunc(abc.ABC, Generic[ScoreT, LabelT]):
  def __init__(self, normal_label: LabelT = 0, outlier_label: LabelT = 1):
    self._normal_label = normal_label
    self._outlier_label = outlier_label

  @property
  @abc.abstractmethod
  def is_stateful(self) -> bool:
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def threshold(self) -> Optional[ScoreT]:
    raise NotImplementedError

  def __call__(self, score: ScoreT) -> LabelT:
    raise NotImplementedError


class AggregationFunc(abc.ABC, Generic[ScoreT, LabelT]):
  @abc.abstractmethod
  def __call__(
      self, predictions: Iterable[AnomalyPrediction[ScoreT, LabelT]]
  ) -> AnomalyPrediction[ScoreT, LabelT]:
    raise NotImplementedError
