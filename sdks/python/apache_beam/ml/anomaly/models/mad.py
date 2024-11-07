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

import apache_beam as beam
from apache_beam.ml.anomaly import univariate
from apache_beam.ml.anomaly.base import AnomalyModel
from apache_beam.ml.anomaly.univariate import EPSILON


class MedianAbsoluteDeviation(AnomalyModel[beam.Row, float]):
  def __init__(self, window_size=10, scale_factor=0.67449):
    self._median_tracker = univariate.SimpleMedianTracker(window_size)
    self._mad_tracker = univariate.SimpleMADTracker(window_size)
    self._scale_factor = scale_factor

  def learn_one(self, x: beam.Row) -> None:
    assert len(x.__dict__) == 1, "MAD requires univariate input"
    v = next(iter(x))
    self._median_tracker.push(v)
    self._mad_tracker.push(v)

  def score_one(self, x: beam.Row) -> float:
    assert len(x.__dict__) == 1, "MAD requires univariate input"
    v = next(iter(x))
    median = self._median_tracker.get()
    mad = self._mad_tracker.get()
    if mad < EPSILON:
      return float('NaN')
    return abs((v - median) / mad * self._scale_factor)
