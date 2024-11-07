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

import collections
import itertools
import math
import statistics
from typing import Any
from typing import Iterable

from apache_beam.ml.anomaly.base import AnomalyPrediction
from apache_beam.ml.anomaly.base import AggregationFunc


class LabelAggregation(AggregationFunc[Any, int]):

  def __call__(
      self, predictions: Iterable[AnomalyPrediction[Any, int]]
  ) -> AnomalyPrediction[Any, int]:
    labels = list(
        itertools.filterfalse(
            lambda label: label is None,
            map(lambda prediction: prediction.label, predictions)))

    if len(labels) == 0:
      return AnomalyPrediction(model_id=self._model_override)

    label = self._agg_func(labels)  # type: ignore

    history = list(predictions) if self._include_history else None

    return AnomalyPrediction(
        model_id=self._model_override, label=label, agg_history=history)


class ScoreAggregation(AggregationFunc[float, Any]):

  def __call__(
      self, predictions: Iterable[AnomalyPrediction[float, Any]]
  ) -> AnomalyPrediction[float, Any]:
    scores = list(
        itertools.filterfalse(
            lambda score: score is None or math.isnan(score),
            map(lambda prediction: prediction.score, predictions)))

    if len(scores) == 0:
      return AnomalyPrediction(model_id=self._model_override)

    score = self._agg_func(scores)  # type: ignore

    history = list(predictions) if self._include_history else None

    return AnomalyPrediction(
        model_id=self._model_override, score=score, agg_history=history)


def MajorityVote(normal_label=0,
                 outlier_label=1,
                 tie_breaker=0,
                 **kwargs) -> AggregationFunc[Any, int]:

  def inner(predictions: Iterable[int]) -> int:
    counters = collections.Counter(predictions)
    if counters[normal_label] < counters[outlier_label]:
      vote = outlier_label
    elif counters[normal_label] > counters[outlier_label]:
      vote = normal_label
    else:
      vote = tie_breaker
    return vote

  return LabelAggregation(inner, **kwargs)


# And scheme
def AllVote(normal_label=0,
            outlier_label=1,
            **kwargs) -> AggregationFunc[Any, int]:

  def inner(predictions: Iterable[int]) -> int:
    return outlier_label if all(map(lambda p: p == outlier_label,
                                    predictions)) else normal_label

  return LabelAggregation(inner, **kwargs)


# Or scheme
def AnyVote(normal_label=0,
            outlier_label=1,
            **kwargs) -> AggregationFunc[Any, int]:

  def inner(predictions: Iterable[int]) -> int:
    return outlier_label if any(map(lambda p: p == outlier_label,
                                    predictions)) else normal_label

  return LabelAggregation(inner, **kwargs)


def AverageScore() -> AggregationFunc[float, Any]:
  return ScoreAggregation(statistics.mean)
