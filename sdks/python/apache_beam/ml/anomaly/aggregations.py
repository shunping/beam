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
import math
import statistics
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Union

from apache_beam.ml.anomaly.base import AnomalyPrediction
from apache_beam.ml.anomaly.base import AggregationFunc
from apache_beam.ml.anomaly.base import LabelT
from apache_beam.ml.anomaly.base import ScoreT


class SimpleAggregation(AggregationFunc[ScoreT, LabelT]):
  def __init__(
      self,
      agg_func: Union[Callable[[Iterable[ScoreT]], ScoreT],
                      Callable[[Iterable[LabelT]], LabelT]],
      include_history: bool = False,
      model_override: Optional[str] = None):
    self._agg_func = agg_func
    self._include_history = include_history
    self._model_override = model_override


class LabelAggregation(SimpleAggregation[Any, LabelT]):
  def __call__(
      self, predictions: Iterable[AnomalyPrediction[Any, LabelT]]
  ) -> AnomalyPrediction[Any, LabelT]:
    labels = [
        prediction.label for prediction in predictions
        if prediction.label is not None
    ]

    if len(labels) == 0:
      return AnomalyPrediction(model_id=self._model_override)

    label = self._agg_func(labels)

    history = list(predictions) if self._include_history else None

    return AnomalyPrediction(
        model_id=self._model_override, label=label, agg_history=history)


class ScoreAggregation(SimpleAggregation[ScoreT, Any]):
  def __call__(
      self, predictions: Iterable[AnomalyPrediction[ScoreT, Any]]
  ) -> AnomalyPrediction[ScoreT, Any]:
    scores = [
        prediction.score
        for prediction in predictions
        if prediction.score is not None and
        not math.isnan(prediction.score)  # type: ignore
    ]

    if len(scores) == 0:
      return AnomalyPrediction(model_id=self._model_override)

    score = self._agg_func(scores)

    history = list(predictions) if self._include_history else None

    return AnomalyPrediction(
        model_id=self._model_override, score=score, agg_history=history)


def MajorityVote(
    normal_label=0,
    outlier_label=1,
    tie_breaker=0,
    **kwargs) -> LabelAggregation:
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
def AllVote(normal_label=0, outlier_label=1, **kwargs) -> LabelAggregation[int]:
  def inner(predictions: Iterable[int]) -> int:
    return outlier_label if all(
        map(lambda p: p == outlier_label, predictions)) else normal_label

  return LabelAggregation(inner, **kwargs)


# Or scheme
def AnyVote(normal_label=0, outlier_label=1, **kwargs) -> LabelAggregation[int]:
  def inner(predictions: Iterable[int]) -> int:
    return outlier_label if any(
        map(lambda p: p == outlier_label, predictions)) else normal_label

  return LabelAggregation(inner, **kwargs)


def AverageScore(**kwargs) -> ScoreAggregation[float]:
  return ScoreAggregation(statistics.mean, **kwargs)


def MaxScore(**kwargs) -> ScoreAggregation[float]:
  return ScoreAggregation(max, **kwargs)
