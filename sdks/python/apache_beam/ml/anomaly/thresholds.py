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

import dataclasses
from typing import Any
from typing import cast
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Union

import apache_beam as beam
from apache_beam.coders import DillCoder
from apache_beam.ml.anomaly import univariate
from apache_beam.ml.anomaly.base import AnomalyResult
from apache_beam.ml.anomaly.base import ThresholdFn
from apache_beam.ml.anomaly.configurable import Config
from apache_beam.ml.anomaly.configurable import configurable
from apache_beam.ml.anomaly.configurable import Configurable
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec
from apache_beam.transforms.userstate import ReadModifyWriteRuntimeState


class BaseThresholdDoFn(beam.DoFn):
  def __init__(self, threshold_fn_config: Config):
    self._threshold_fn_config = threshold_fn_config
    self._threshold_fn: ThresholdFn

  def _update_prediction(self, result: AnomalyResult) -> AnomalyResult:
    label = self._threshold_fn.apply(result.prediction.score)
    return dataclasses.replace(
        result,
        prediction=dataclasses.replace(
            result.prediction,
            label=label,
            threshold=self._threshold_fn.threshold))


class StatelessThresholdDoFn(BaseThresholdDoFn):
  def __init__(self, threshold_fn_config: Config):
    threshold_fn_config.args["_run_init"] = True
    self._threshold_fn = cast(
        ThresholdFn, Configurable.from_config(threshold_fn_config))
    assert not self._threshold_fn.is_stateful, \
      "This DoFn can only take stateless function as threshold_fn"

  def process(self, element: Tuple[Any, Tuple[Any, AnomalyResult]],
              **kwargs) -> Iterable[Tuple[Any, Tuple[Any, AnomalyResult]]]:
    k1, (k2, prediction) = element
    yield k1, (k2, self._update_prediction(prediction))


class StatefulThresholdDoFn(BaseThresholdDoFn):
  THRESHOLD_STATE_INDEX = ReadModifyWriteStateSpec('saved_tracker', DillCoder())

  def __init__(self, threshold_fn_config: Config):
    threshold_fn_config.args["_run_init"] = True
    threshold_fn: ThresholdFn = cast(
        ThresholdFn, Configurable.from_config(threshold_fn_config))
    assert threshold_fn.is_stateful, \
      "This DoFn can only take stateful function as threshold_fn"
    self._threshold_fn_config = threshold_fn_config

  def process(
      self,
      element: Tuple[Any, Tuple[Any, AnomalyResult]],
      threshold_state: Union[ReadModifyWriteRuntimeState,
                             Any] = beam.DoFn.StateParam(THRESHOLD_STATE_INDEX),
      **kwargs) -> Iterable[Tuple[Any, Tuple[Any, AnomalyResult]]]:
    k1, (k2, prediction) = element

    self._threshold_fn = threshold_state.read()
    if self._threshold_fn is None:
      self._threshold_fn: Configurable = Configurable.from_config(
          self._threshold_fn_config)

    yield k1, (k2, self._update_prediction(prediction))

    threshold_state.write(self._threshold_fn)


@configurable
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
    if score is None or score < self.threshold:
      return self._normal_label

    return self._outlier_label


@configurable
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
