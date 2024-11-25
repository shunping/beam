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
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar

import apache_beam as beam

ConfigT = TypeVar('ConfigT', bound='Configurable')


@dataclasses.dataclass(frozen=True)
class Config():
  type: str
  args: dict[str, Any] = dataclasses.field(default_factory=dict)


class Configurable():
  _skip_list = ["self"]
  _known_subclasses = {}  # a mutable class variable, sharing by all subclasses
  _key = None

  @classmethod
  def register(cls, name, subclass, error_if_exists=True) -> None:
    if name in cls._known_subclasses and error_if_exists:
      raise ValueError(f"{name} is already registered for {subclass.__name__}")
    cls._known_subclasses[name] = subclass
    subclass._key = name

  @classmethod
  def unregister(cls, name) -> None:
    if name in cls._known_subclasses:
      del cls._known_subclasses[name]

  @staticmethod
  def _from_config_helper(v):
    if isinstance(v, Config):
      return Configurable.from_config(v)

    if isinstance(v, List):
      return [Configurable._from_config_helper(e) for e in v]

    return v

  # TODO: change to Self (PEP 673) after the minimum python support of Beam
  # reaches 3.11.
  # Refer to https://github.com/python/typing/issues/58
  @classmethod
  def from_config(cls: Type[ConfigT], config: Config) -> ConfigT:
    if config is None:
      raise ValueError("Config cannot be None")

    if config.type is None:
      raise ValueError(f"Type not found in config {config}")

    subclass = cls._known_subclasses.get(config.type, None)
    if subclass is None:
      raise ValueError(f"Unknown config type in config {config}")

    args = {k:Configurable._from_config_helper(v) for k,v in config.args.items()}
    return subclass(**args)

  @staticmethod
  def _to_config_helper(v):
    v_class = v.__class__
    if issubclass(v_class, Configurable):
      return v.to_config()

    if issubclass(v_class, List):
      return [Configurable._to_config_helper(e) for e in v]

    return v

  def to_config(self) -> Config:
    if self.__class__._key is None:
      raise ValueError(f"Class {self.__class__.__name__} is not registered.")

    args = []
    for cls in self.__class__.mro():
      args.extend(inspect.getfullargspec(cls.__init__).args)

    config_args = {}
    for arg in set(args):
      if arg in self._skip_list:
        continue

      if hasattr(self, f"_{arg}"):
        v = getattr(self, f"_{arg}")
        config_args[arg] = Configurable._to_config_helper(v)
      else:
        logging.debug("_%s not found in the attributes of object %s", arg,
                      self.__class__.__name__)

    ret = Config(type=self.__class__._key, args=config_args)
    return ret


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


class AnomalyDetector(abc.ABC, Configurable):

  def __init__(self,
               model_id: Optional[str] = None,
               features: Optional[Iterable[str]] = None,
               target: Optional[str] = None,
               threshold_criterion: Optional[ThresholdFn] = None,
               initialize_model=False,
               **kwargs):
    self._model_id = model_id if model_id is not None else self._key
    self._features = features
    self._target = target
    self._threshold_criterion = threshold_criterion
    self._init_model = initialize_model

  @abc.abstractmethod
  def learn_one(self, x: beam.Row) -> None:
    raise NotImplementedError

  @abc.abstractmethod
  def score_one(self, x: beam.Row) -> float:
    raise NotImplementedError


class EnsembleAnomalyDetector(AnomalyDetector):

  def __init__(self,
               n: int = 10,
               aggregation_strategy: Optional[AggregationFn] = None,
               learners: Optional[List[AnomalyDetector]] = None,
               **kwargs):
    self._n = n
    self._aggregation_strategy = aggregation_strategy
    self._learners = learners
    if self._learners:
      self._n = len(self._learners)

    if "model_id" not in kwargs:
      kwargs["model_id"] = self._key if self._key is not None else "custom"

    super().__init__(**kwargs)

  def learn_one(self, x: beam.Row) -> None:
    raise NotImplementedError

  def score_one(self, x: beam.Row) -> float:
    raise NotImplementedError

EnsembleAnomalyDetector.register("custom", EnsembleAnomalyDetector)
