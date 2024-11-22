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
import dataclasses
from dataclasses import dataclass
import inspect
import logging
from typing import Any
from typing import Iterable
from typing import Optional

import apache_beam as beam


@dataclass(frozen=True)
class AnomalyPrediction():
  model_id: Optional[str] = None
  score: Optional[float] = None
  label: Optional[int] = None
  threshold: Optional[float] = None
  info: str = ""
  agg_history: Optional[Iterable[AnomalyPrediction]] = None


@dataclass(frozen=True)
class AnomalyResult():
  example: beam.Row
  prediction: AnomalyPrediction


class Configurable():
  _known_subclasses = {}
  _key = None

  @classmethod
  def register(cls, name, subclass) -> None:
    cls._known_subclasses[name] = subclass
    subclass._key = name

  @classmethod
  def from_config(cls, config) -> Optional[Configurable]:
    if config is None:
      return None

    if "type" not in config:
      raise NotImplementedError(f"Type not found in config {config}")

    subclass = cls._known_subclasses.get(config["type"], None)
    if subclass is None:
      raise NotImplementedError(f"Unknown config type in config {config}")

    config.pop("type")
    return subclass(**config)

  def to_config(self) -> dict[str, Any]:
    if self.__class__._key is None:
      raise ValueError(f"Class {self.__class__.__name__} is not registered.")

    config = {"type": self.__class__._key}

    args = []
    for cls in self.__class__.mro():
      args.extend(inspect.getfullargspec(cls.__init__).args)

    for arg in set(args):
      if hasattr(self, f"_{arg}"):
        config[arg] = getattr(self, f"_{arg}")
      else:
        logging.warning("Unable to find _%s in the object of %s", arg,
                        self.__class__.__name__)

    return config


class ThresholdFn(abc.ABC, Configurable):

  def __init__(self, normal_label: int = 0, outlier_label: int = 1):
    self._normal_label = normal_label
    self._outlier_label = outlier_label

  @property
  @abc.abstractmethod
  def is_stateful(self) -> bool:
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def threshold(self) -> Optional[float]:
    raise NotImplementedError

  @abc.abstractmethod
  def apply(self, score: Optional[float]) -> int:
    raise NotImplementedError


class AggregationFn(abc.ABC, Configurable):

  @abc.abstractmethod
  def apply(self,
            predictions: Iterable[AnomalyPrediction]) -> AnomalyPrediction:
    raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class AnomalyDetectorConfig():
  algorithm: str
  algorithm_args: Optional[dict[str, Any]] = None
  model_id: Optional[str] = None
  features: Optional[Iterable[str]] = None
  target: Optional[str] = None
  threshold_criterion: Optional[dict[str, Any]] = None

  # def __post_init__(self):
  #   canonical_alg = self.algorithm.lower()
  #   if canonical_alg not in KNOWN_ALGORITHMS:
  #     raise NotImplementedError(f"Algorithm '{self.algorithm}' not found")

  #   if not self.model_id:
  #     object.__setattr__(self, 'model_id', self.algorithm)


class AnomalyDetector(abc.ABC):

  def __init__(self,
               model_id: Optional[str] = None,
               features: Optional[Iterable[str]] = None,
               target: Optional[str] = None,
               threshold_criterion: Optional[ThresholdFn] = None,
               initialize_model=False):
    self._model_id = model_id
    self._features = features
    self._target = target
    self._threshold_criterion = threshold_criterion
    self._initialize_model = initialize_model

  @abc.abstractmethod
  def learn_one(self, x: beam.Row) -> None:
    raise NotImplementedError

  @abc.abstractmethod
  def score_one(self, x: beam.Row) -> float:
    raise NotImplementedError
