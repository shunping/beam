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

import math

import apache_beam as beam
from apache_beam.ml.anomaly import univariate
from apache_beam.ml.anomaly.base import AnomalyDetector
from apache_beam.ml.anomaly.configurable import configurable
from apache_beam.ml.anomaly.univariate import EPSILON


@configurable(key="sad", lazy_init=False)
class StandardAbsoluteDeviation(AnomalyDetector):
  def __init__(self, sub_stat="mean", window_size=10, **kwargs):
    super().__init__(**kwargs)
    self._window_size = window_size
    self._sub_stat = sub_stat

    self._sub_stat_tracker = None
    self._stdev_tracker = None

    if self._init_model:
      self.initialize()

  def initialize(self):
    if self._sub_stat == 'mean':
      self._sub_stat_tracker = univariate.SimpleMeanTracker(self._window_size)
    elif self._sub_stat == 'median':
      self._sub_stat_tracker = univariate.SimpleMedianTracker(self._window_size)
    else:
      raise ValueError(f"unknown sub_stat {self._sub_stat}")

    self._stdev_tracker = univariate.SimpleStdevTracker(self._window_size)

  def learn_one(self, x: beam.Row) -> None:
    assert len(x.__dict__) == 1, "SAD requires univariate input"
    assert self._sub_stat_tracker is not None
    assert self._stdev_tracker is not None

    v = next(iter(x))
    self._stdev_tracker.push(v)
    self._sub_stat_tracker.push(v)

  def score_one(self, x: beam.Row) -> float:
    assert self._sub_stat_tracker is not None
    assert self._stdev_tracker is not None

    assert len(x.__dict__) == 1, "SAD requires univariate input"
    v = next(iter(x))
    sub_stat = self._sub_stat_tracker.get()
    stdev = self._stdev_tracker.get()
    if math.isnan(stdev) or abs(stdev) < EPSILON:
      return 0.0
    return abs((v - sub_stat) / stdev)
