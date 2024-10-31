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

from apache_beam.ml.anomaly.base import AnomalyDecision


def majority_vote(outlier_label=1, normal_label=0, tie_breaker=0, include_history=False):

  def get_vote(decisions: Iterable[AnomalyDecision]) -> AnomalyDecision:
    predictions = itertools.filterfalse(
        lambda prediction: prediction is not None,
        map(lambda decision: decision.prediction, decisions))

    counters = collections.Counter(predictions)
    if len(counters) == 0:
      return AnomalyDecision(model="aggregate", prediction=None)

    if counters[normal_label] < counters[outlier_label]:
      vote = outlier_label
    elif counters[normal_label] > counters[outlier_label]:
      vote = normal_label
    else:
      vote = tie_breaker
    info=('[' + ('; '.join(map(str, decisions))) + ']') if include_history else ''

    return AnomalyDecision(
        model="aggregate",
        prediction=vote,
        info=info)

  return get_vote


# AND scheme
def all_vote(outlier_label=1, normal_label=0, include_history=False):

  def get_vote(decisions: Iterable[AnomalyDecision]) -> AnomalyDecision:
    predictions = itertools.filterfalse(
        lambda prediction: prediction is None,
        map(lambda decision: decision.prediction, decisions))

    vote = outlier_label if all(map(lambda p: p == outlier_label,
                                    predictions)) else normal_label
    info=('[' + ('; '.join(map(str, decisions))) + ']') if include_history else ''

    return AnomalyDecision(
        model="aggregate",
        prediction=vote,
        info=info)

  return get_vote


# OR scheme
def any_vote(outlier_label=1, normal_label=0, include_history=False):

  def get_vote(decisions: Iterable[AnomalyDecision]):
    predictions = list(
        itertools.filterfalse(
            lambda prediction: prediction is None,
            map(lambda decision: decision.prediction, decisions)))

    vote = outlier_label if any(map(lambda p: p == outlier_label,
                                    predictions)) else normal_label
    info=('[' + ('; '.join(map(str, decisions))) + ']') if include_history else ''

    return AnomalyDecision(
        model="aggregate",
        prediction=vote,
        info=info)

  return get_vote

def average_score(include_history=False):
  def get_score(decisions: Iterable[AnomalyDecision]) -> AnomalyDecision:
    scores: Iterable[float] = itertools.filterfalse(
        lambda score: math.isnan(score),
        map(lambda decision: decision.score, decisions))
    info=('[' + ('; '.join(map(str, decisions))) + ']') if include_history else ''

    return AnomalyDecision(model="aggregate", score=statistics.mean(scores), info=info)

  return get_score
