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

import numpy as np

import apache_beam as beam
from apache_beam.ml.anomaly.base import BaseAnomalyModel
from apache_beam.ml.anomaly import univariate

class LodaWeakLearner(BaseAnomalyModel):

  def __init__(self, n_init=256, histogram_tracker=None):
    if histogram_tracker is None:
      self._hist = univariate.SimpleHistogram(window_size=256, n_bins=256)
    else:
      self._hist = histogram_tracker

    self._n_init = n_init
    self._features = None
    self._projection = None

  def learn_one(self, x: beam.Row):
    if self._features is None:
      self._features = sorted(x.__dict__.keys())

    if self._projection is None:
      n_features = len(self._features)
      self._projection = np.random.randn(n_features)
      n_nonzero_dims = int(np.sqrt(n_features))
      zero_idx = np.random.permutation(len(self._features))[:(n_features -
                                                              n_nonzero_dims)]
      self._projection[zero_idx] = 0

    x_np = np.array([x.__dict__[k] for k in self._features])
    projected_data = x_np.dot(self._projection)
    self._hist.push(projected_data)

  def score_one(self, x: beam.Row):
    if len(
        self._hist._queue
    ) < self._n_init or self._features is None or self._projection is None:
      y_pred = 0
    else:
      x_np = np.array([x.__dict__[k] for k in self._features])
      projected_data = x_np.dot(self._projection)

      histogram, limits = self._hist.get()
      histogram = histogram.astype(np.float64)
      histogram += 1e-12
      histogram /= np.sum(histogram)

      inds = np.searchsorted(limits[:256 - 1], projected_data, side='left')
      y_pred = -np.log(histogram[inds])

    return y_pred
