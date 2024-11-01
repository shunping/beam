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

from collections import deque
import math

import numpy as np


class BaseTracker:
  def push(self, x):
    raise NotImplementedError()

  def get(self):
    raise NotImplementedError()


class BatchTracker(BaseTracker):
  def __init__(self, window_size):
    self._queue = deque(maxlen=window_size)

  def push(self, x):
    self._queue.append(x)


class SimpleMeanTracker(BatchTracker):
  def get(self):
    # TODO: check if we should return 0 or NaN.
    if len(self._queue) == 0:
      return float('NaN')
    return np.nanmean(self._queue)


class SimpleStdevTracker(BatchTracker):
  def get(self):
    # We do not use nanstd, since nanstd([]) gives 0, which is incorrect.
    # Use nanvar instead.
    return math.sqrt(np.nanvar(self._queue, ddof=1))


class SimpleMedianTracker(BatchTracker):
  def get(self):
    return np.nanmedian(self._queue)


class SimpleMADTracker(BatchTracker):
  def get(self):
    median = np.nanmedian(self._queue)
    return np.nanmedian([
        abs(x - median) if not math.isnan(x) else float('nan')
        for x in self._queue
    ])


class SimpleHistogram(BatchTracker):
  def __init__(self, window_size, n_bins):
    super().__init__(window_size)
    self._n_bins = n_bins

  def get(self):
    return np.histogram(self._queue, bins=self._n_bins, density=False)


class SimpleQuantile(BatchTracker):
  def __init__(self, window_size, quantile):
    assert 0 < quantile < 1, "quantile argument should be between 0 and 1"

    super().__init__(window_size)
    self._quantile = quantile

  def get(self):
    return np.nanquantile(self._queue, self._quantile)


class RollingTracker(BaseTracker):
  def __init__(self, window_size):
    self._window_size = window_size
    self._len = 0
    self._queue = deque(maxlen=window_size + 1)

  def push(self, x):
    self._queue.append(x)

  def pop(self):
    return self._queue.popleft()


class RollingMeanTracker(RollingTracker):
  # This is an modified version of https://en.wikipedia.org/wiki/Moving_average,
  # that takes care of NaN within the window.

  def __init__(self, window_size):
    super().__init__(window_size)
    self._mean = 0
    self._n = 0

  def push(self, x):
    super().push(x)

    if not math.isnan(x):
      self._n += 1
      delta1 = x - self._mean
    else:
      delta1 = 0

    if (len(self._queue) > self._window_size and
        not math.isnan(old_x := super().pop())):
      self._n -= 1
      delta2 = self._mean - old_x
    else:
      delta2 = 0

    if self._n > 0:
      self._mean += (delta1 + delta2) / self._n
    else:
      self._mean = 0

  def get(self):
    if self._n < 1:
      return float("nan")

    return self._mean


class RollingStdevTracker(RollingTracker):
  def __init__(self, window_size):
    super().__init__(window_size)
    self._mean = 0
    self._m2 = 0

  def push(self, x):
    if math.isnan(x):
      return

    # self._n += 1
    super().push(x)
    if len(self._queue) <= self._window_size:
      # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford%27s_online_algorithm
      delta = x - self._mean
      self._mean += delta / len(self._queue)
      delta2 = x - self._mean
      self._m2 += delta * delta2
    else:
      # Refer to https://jonisalonen.com/2014/efficient-and-accurate-rolling-standard-deviation/
      old_x = super().pop()
      delta = x - old_x
      delta2 = old_x - self._mean
      self._mean += delta / self._window_size
      delta3 = x - self._mean
      self._m2 += delta * delta2 + delta * delta3

  def get(self):
    if len(self._queue) < 2:
      return float("nan")  # keep it consistent with numpy
    dof = min(len(self._queue), self._window_size) - 1
    return math.sqrt(self._m2 / dof)
