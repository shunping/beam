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

from __future__ import annotations

import copy
import dataclasses
from typing import Optional

import apache_beam as beam
from apache_beam.coders import DillCoder
from apache_beam.ml.anomaly import univariate
from apache_beam.ml.anomaly.base import AnomalyResult
from apache_beam.ml.anomaly.base import ThresholdFn
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec


class BaseThresholdDoFn(beam.DoFn):

  def __init__(self, threshold_func: ThresholdFn):
    self._threshold_func = threshold_func

  def _update_prediction(self, result: AnomalyResult) -> AnomalyResult:
    label = self._threshold_func.apply(result.prediction.score)
    return dataclasses.replace(
        result,
        prediction=dataclasses.replace(
            result.prediction,
            label=label,
            threshold=self._threshold_func.threshold))


class StatelessThresholdDoFn(BaseThresholdDoFn):

  def __init__(self, threshold_func: ThresholdFn):
    assert not threshold_func.is_stateful, \
      "This DoFn can only take stateless function as threshold_func"
    self._threshold_func = threshold_func

  def process(self, element,
              **kwargs):
    k1, (k2, prediction) = element
    yield k1, (k2, self._update_prediction(prediction))


class StatefulThresholdDoFn(BaseThresholdDoFn):
  THRESHOLD_STATE_INDEX = ReadModifyWriteStateSpec('saved_tracker', DillCoder())

  def __init__(self, threshold_func: ThresholdFn):
    assert threshold_func.is_stateful, \
      "This DoFn can only take stateful function as threshold_func"
    self._original_func = threshold_func

  def process(self,
              element,
              threshold_state=beam.DoFn.StateParam(THRESHOLD_STATE_INDEX),
              **kwargs):
    k1, (k2, prediction) = element

    self._threshold_func = threshold_state.read()  # type: ignore
    if self._threshold_func is None:
      self._threshold_func = copy.deepcopy(self._original_func)

    yield k1, (k2, self._update_prediction(prediction))

    threshold_state.write(self._threshold_func)  # type: ignore


class FixedThreshold(ThresholdFn):

  def __init__(self, cutoff: float, **kwargs):
    super().__init__(**kwargs)
    self._cutoff = cutoff

  @property
  def is_stateful(self) -> bool:
    return False

  @property
  def threshold(self) -> float:
    return self._cutoff

  def apply(self, score: Optional[float]) -> int:
    if score is None or score < self.threshold:  # type: ignore
      return self._normal_label

    return self._outlier_label


ThresholdFn.register("fixed", FixedThreshold)


class QuantileThreshold(ThresholdFn):

  def __init__(self, quantile: float, **kwargs):
    super().__init__(**kwargs)
    self._quantile = quantile
    self._tracker_class = univariate.SimpleQuantile
    self._tracker_kwargs = {"window_size": 100}
    self._tracker = self._tracker_class(**self._tracker_kwargs)

  @property
  def is_stateful(self) -> bool:
    return True

  @property
  def threshold(self) -> float:
    return self._tracker.get(self._quantile)

  def apply(self, score: Optional[float]) -> int:
    self._tracker.push(score)

    if score is None or score < self.threshold:
      return self._normal_label

    return self._outlier_label


ThresholdFn.register("quantile", QuantileThreshold)
