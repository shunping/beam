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
from typing import Iterable
from typing import Callable

from apache_beam.ml.anomaly.base import AnomalyPrediction
from apache_beam.ml.anomaly.base import BaseAggregationFunc


class LabelAggregation(BaseAggregationFunc[int]):
  def __call__(self,
               decisions: Iterable[AnomalyPrediction]) -> AnomalyPrediction:
    labels = list(
        itertools.filterfalse(lambda prediction: prediction is None,
                              map(lambda decision: decision.label, decisions)))

    if len(labels) == 0:
      return AnomalyPrediction(model_id=self._model_override)

    prediction = self._agg_func(labels, **self._kwargs)  # type: ignore

    info = ('[' + ('; '.join(map(str, decisions))) +
            ']') if self._include_history else ''

    return AnomalyPrediction(
        model_id=self._model_override, label=prediction, info=info)


class ScoreAggregation(BaseAggregationFunc[float]):
  def __call__(self,
               decisions: Iterable[AnomalyPrediction]) -> AnomalyPrediction:
    scores = list(
        itertools.filterfalse(lambda score: score is None or math.isnan(score),
                              map(lambda decision: decision.score, decisions)))

    if len(scores) == 0:
      return AnomalyPrediction(model_id=self._model_override)

    score = self._agg_func(scores, **self._kwargs)  # type: ignore

    info = ('[' + ('; '.join(map(str, decisions))) +
            ']') if self._include_history else ''

    return AnomalyPrediction(
        model_id=self._model_override, score=score, info=info)


def MajorityVote(normal_label=0,
                 outlier_label=1,
                 tie_breaker=0) -> Callable[[Iterable[int]], int]:

  def inner(predictions: Iterable[int]) -> int:
    counters = collections.Counter(predictions)
    if counters[normal_label] < counters[outlier_label]:
      vote = outlier_label
    elif counters[normal_label] > counters[outlier_label]:
      vote = normal_label
    else:
      vote = tie_breaker
    return vote

  return inner


# And scheme
def AllVote(normal_label=0, outlier_label=1) -> Callable[[Iterable[int]], int]:

  def inner(predictions: Iterable[int]) -> int:
    return outlier_label if all(map(lambda p: p == outlier_label,
                                    predictions)) else normal_label

  return inner


# Or scheme
def AnyVote(normal_label=0, outlier_label=1) -> Callable[[Iterable[int]], int]:

  def inner(predictions: Iterable[int]) -> int:
    return outlier_label if any(map(lambda p: p == outlier_label,
                                    predictions)) else normal_label

  return inner


def AverageScore(scores: Iterable[float]) -> float:
  return statistics.mean(scores)
