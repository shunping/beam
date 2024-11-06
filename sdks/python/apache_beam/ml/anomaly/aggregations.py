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

from apache_beam.ml.anomaly.base import AnomalyPrediction
from apache_beam.ml.anomaly.base import BaseAggregation


class PredictionAggregation(BaseAggregation):

  def __init__(self,
               outlier_label=1,
               normal_label=0,
               include_history=False,
               model_override=""):
    self._outlier_label = outlier_label
    self._normal_label = normal_label
    self._include_history = include_history
    self._model_override = model_override

  def aggregate_predictions(self, predictions: Iterable[int]):
    raise NotImplementedError

  def __call__(self, decisions: Iterable[AnomalyPrediction]) -> AnomalyPrediction:
    predictions = list(
        itertools.filterfalse(
            lambda prediction: prediction is None,
            map(lambda decision: decision.label, decisions)))

    if len(predictions) == 0:
      return AnomalyPrediction(model_id=self._model_override)

    prediction = self.aggregate_predictions(predictions)  # type: ignore

    info = ('[' + ('; '.join(map(str, decisions))) +
            ']') if self._include_history else ''

    return AnomalyPrediction(
        model_id=self._model_override, label=prediction, info=info)


class MajorityVote(PredictionAggregation):

  def __init__(self, tie_breaker=0, **kwargs):
    self._tie_breaker = tie_breaker
    super().__init__(**kwargs)

  def aggregate_predictions(self, predictions: Iterable[int]) -> int:
    counters = collections.Counter(predictions)

    if counters[self._normal_label] < counters[self._outlier_label]:
      vote = self._outlier_label
    elif counters[self._normal_label] > counters[self._outlier_label]:
      vote = self._normal_label
    else:
      vote = self._tie_breaker

    return vote

# And scheme
class AllVote(PredictionAggregation):
  def aggregate_predictions(self, predictions: Iterable[int]) -> int:
    return self._outlier_label if all(map(lambda p: p == self._outlier_label,
                                    predictions)) else self._normal_label


# Or scheme
class AnyVote(PredictionAggregation):
  def aggregate_predictions(self, predictions: Iterable[int]) -> int:
    return self._outlier_label if any(map(lambda p: p == self._outlier_label,
                                predictions)) else self._normal_label


# class ScoreAggregation(BaseAggregation):

#   def __init__(self,
#                include_history=False,
#                model_override=""):
#     self._include_history = include_history
#     self._model_override = model_override

#   def aggregate_scores(self, predictions: Iterable[float]):
#     raise NotImplementedError

#   def __call__(self, decisions: Iterable[AnomalyPrediction]) -> AnomalyPrediction:
#     scores = list(itertools.filterfalse(
#         lambda score: math.isnan(score),
#         map(lambda decision: decision.score, decisions)))

#     if len(scores) == 0:
#       return AnomalyPrediction(model_id=self._model_override)

#     score = self.aggregate_scores(scores)  # type: ignore

#     info = ('[' + ('; '.join(map(str, decisions))) +
#             ']') if self._include_history else ''

#     return AnomalyPrediction(
#         model_id=self._model_override, score=score, info=info)


# class AverageScore(ScoreAggregation):
#   def aggregate_scores(self, scores: Iterable[float]) -> float:
#     return statistics.mean(scores)

def AverageScore(scores: Iterable[float]) -> float:
  return statistics.mean(scores)
