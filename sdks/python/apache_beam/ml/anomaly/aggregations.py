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
from typing import Callable
from typing import Iterable

from apache_beam.ml.anomaly.base import AnomalyPrediction
from apache_beam.ml.anomaly.base import AggregationFn
from apache_beam.ml.anomaly.base import configurable
from apache_beam.ml.anomaly.base import register_configurable


class LabelAggregation(AggregationFn):

  def __init__(self,
               agg_func: Callable[[Iterable[int]], int],
               include_history: bool = False):
    self._agg = agg_func
    self._include_history = include_history
    self._model_override = None

  def apply(self,
            predictions: Iterable[AnomalyPrediction]) -> AnomalyPrediction:
    labels = [
        prediction.label
        for prediction in predictions
        if prediction.label is not None
    ]

    if len(labels) == 0:
      return AnomalyPrediction(model_id=self._model_override)

    label = self._agg(labels)

    history = list(predictions) if self._include_history else None

    return AnomalyPrediction(
        model_id=self._model_override, label=label, agg_history=history)


class ScoreAggregation(AggregationFn):

  def __init__(self,
               agg_func: Callable[[Iterable[float]], float],
               include_history: bool = False):
    self._agg = agg_func
    self._include_history = include_history
    self._model_override = None

  def apply(self,
            predictions: Iterable[AnomalyPrediction]) -> AnomalyPrediction:
    scores = [
        prediction.score
        for prediction in predictions
        if prediction.score is not None and
        not math.isnan(prediction.score)  # type: ignore
    ]

    if len(scores) == 0:
      return AnomalyPrediction(model_id=self._model_override)

    score = self._agg(scores)

    history = list(predictions) if self._include_history else None

    return AnomalyPrediction(
        model_id=self._model_override, score=score, agg_history=history)

@configurable
class MajorityVote(LabelAggregation):

  def __init__(self, normal_label=0, outlier_label=1, tie_breaker=0, **kwargs):
    self._tie_breaker = tie_breaker
    self._normal_label = normal_label
    self._outlier_label = outlier_label

    def inner(predictions: Iterable[int]) -> int:
      counters = collections.Counter(predictions)
      if counters[self._normal_label] < counters[self._outlier_label]:
        vote = self._outlier_label
      elif counters[self._normal_label] > counters[self._outlier_label]:
        vote = self._normal_label
      else:
        vote = self._tie_breaker
      return vote

    super().__init__(agg_func=inner, **kwargs)


register_configurable(MajorityVote, "majority_vote")

# And scheme
@configurable
class AllVote(LabelAggregation):

  def __init__(self, normal_label=0, outlier_label=1, **kwargs):
    self._normal_label = normal_label
    self._outlier_label = outlier_label

    def inner(predictions: Iterable[int]) -> int:
      return self._outlier_label if all(
          map(lambda p: p == self._outlier_label,
              predictions)) else self._normal_label

    super().__init__(agg_func=inner, **kwargs)


register_configurable(AllVote, "all_vote")


# Or scheme
@configurable
class AnyVote(LabelAggregation):

  def __init__(self, normal_label=0, outlier_label=1, **kwargs):
    self._normal_label = normal_label
    self._outlier_label = outlier_label

    def inner(predictions: Iterable[int]) -> int:
      return self._outlier_label if any(
          map(lambda p: p == self._outlier_label,
              predictions)) else self._normal_label

    super().__init__(agg_func=inner, **kwargs)


register_configurable(AnyVote, "any_vote")

@configurable
class AverageScore(ScoreAggregation):

  def __init__(self, **kwargs):
    super().__init__(agg_func=statistics.mean, **kwargs)


register_configurable(AverageScore, "average_score")

@configurable
class MaxScore(ScoreAggregation):

  def __init__(self, **kwargs):
    super().__init__(agg_func=max, **kwargs)


register_configurable(MaxScore, "max_score")
