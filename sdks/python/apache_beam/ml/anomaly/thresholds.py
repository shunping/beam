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

from typing import Any
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Union

import apache_beam as beam
from apache_beam.coders import DillCoder
from apache_beam.ml.anomaly import univariate
from apache_beam.ml.anomaly.base import AnomalyPrediction
from apache_beam.ml.anomaly.base import BaseThresholdFunc
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec


class FixedThreshold(BaseThresholdFunc):

  def __init__(self, threshold: Union[int, float]):
    self._threshold = threshold

  @property
  def threshold(self):
    return self._threshold

  def process(self, element: Tuple[Any, Tuple[Any, AnomalyPrediction]],
              **kwargs) -> Iterable[Tuple[Any, Tuple[Any, AnomalyPrediction]]]:
    k1, (k2, prediction) = element
    yield k1, (k2, self._update_prediction(prediction))


class QuantileThreshold(BaseThresholdFunc):
  TRACKER_STATE_INDEX = ReadModifyWriteStateSpec('saved_tracker', DillCoder())

  def __init__(self,
               quantile: float,
               quantile_tracker_class: Optional[type[
                   univariate.BatchQuantileTracker]] = None,
               quantile_tracker_kwargs: Optional[dict[str, Any]] = None):
    self._quantile = quantile
    if quantile_tracker_class is None:
      self._tracker_class = univariate.SimpleQuantile
      self._tracker_kwargs = {"window_size": 100}
    else:
      self._tracker_class = quantile_tracker_class
      self._tracker_kwargs = quantile_tracker_kwargs if quantile_tracker_kwargs is not None else {}

  @property
  def threshold(self) -> float:
    return self._tracker.get(self._quantile)  # type: ignore

  def process(self,
              element: Tuple[Any, Tuple[Any, AnomalyPrediction]],
              tracker_state=beam.DoFn.StateParam(TRACKER_STATE_INDEX),
              **kwargs) -> Iterable[Tuple[Any, Tuple[Any, AnomalyPrediction]]]:
    k1, (k2, prediction) = element

    self._tracker = tracker_state.read()  # type: ignore
    if self._tracker is None:
      self._tracker = self._tracker_class(**self._tracker_kwargs)
    self._tracker.push(prediction.decision.score)

    yield k1, (k2, self._update_prediction(prediction))

    tracker_state.write(self._tracker)  # type: ignore
