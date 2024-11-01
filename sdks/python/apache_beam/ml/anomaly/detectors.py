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

from typing import Optional
from typing import List
import uuid

from apache_beam.ml.anomaly.base import BaseAggregation
from apache_beam.ml.anomaly.base import BaseThresholdFunc
from apache_beam.ml.anomaly.aggregations import AverageScore

from apache_beam.ml.anomaly.models import KNOWN_ALGORITHMS


class AnomalyDetector:
  def __init__(
      self,
      algorithm: str,
      id: Optional[str] = None,
      features: Optional[List[str]] = None,
      target: Optional[str] = None,
      threshold_func: Optional[BaseThresholdFunc] = None,
      *args,
      **kwargs) -> None:
    algorithm = algorithm.lower()
    if algorithm in KNOWN_ALGORITHMS:
      detector = KNOWN_ALGORITHMS[algorithm]
      if detector is not None:
        self._underlying = detector(*args, **kwargs)
    else:
      raise NotImplementedError(f"algorithm '{algorithm}' not found")

    self._id = id if id else f"{algorithm}_{uuid.uuid4().hex[:6]}"
    self._features = features
    self._target = target
    self._threshold_func = threshold_func

  @property
  def label(self):
    return self._id

  def __repr__(self):
    return self.label

  def score_and_learn(self, x, y, unused_key):
    y_pred = self._underlying.score_one(x)
    self._underlying.learn_one(x)
    return y_pred


class EnsembleAnomalyDetector(AnomalyDetector):
  def __init__(
      self,
      n: int = 10,
      label: Optional[str] = None,
      aggregation_strategy: Optional[BaseAggregation] = AverageScore(),
      **kwargs):
    weak_learner_alg = kwargs["algorithm"]
    kwargs["algorithm"] = "ensemble"
    super().__init__(label=label, **kwargs)

    kwargs["algorithm"] = weak_learner_alg
    self._weak_learners = []
    for _ in range(n):
      self._weak_learners.append(AnomalyDetector(**kwargs))

    self._aggregation_strategy = aggregation_strategy

  def score_and_learn(self, x, y, unused_key):
    raise NotImplementedError()
