
import math

import apache_beam as beam
from apache_beam.ml.anomaly import univariate
from apache_beam.ml.anomaly.base import BaseAnomalyModel
from apache_beam.ml.anomaly.base import EPSILON


class StandardAbsoluteDeviation(BaseAnomalyModel):

  def __init__(self, sub_stat="mean", window_size=10, sub_stat_tracker=None):
    if sub_stat_tracker is None:
      if sub_stat == 'mean':
        self._sub_stat_tracker = univariate.SimpleMeanTracker(window_size)
      elif sub_stat == 'median':
        self._sub_stat_tracker = univariate.SimpleMedianTracker(window_size)
      else:
        raise ValueError(f"unknown sub_stat {sub_stat}")
    else:
      self._sub_stat_tracker = sub_stat_tracker

    self._stdev_tracker = univariate.SimpleStdevTracker(window_size)

  def learn_one(self, x: beam.Row):
    assert len(x.__dict__) == 1, "SAD requires univariate input"
    v = next(iter(x))
    self._stdev_tracker.push(v)
    self._sub_stat_tracker.push(v)

  def score_one(self, x: beam.Row):
    assert len(x.__dict__) == 1, "SAD requires univariate input"
    v = next(iter(x))
    sub_stat = self._sub_stat_tracker.get()
    stdev = self._stdev_tracker.get()
    if math.isnan(stdev) or abs(stdev) < EPSILON:
      return 0.0
    return abs((v - sub_stat) / stdev)