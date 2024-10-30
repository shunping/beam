import apache_beam as beam
from apache_beam.ml.anomaly import univariate
from apache_beam.ml.anomaly.base import BaseAnomalyModel
from apache_beam.ml.anomaly.base import EPSILON

class MedianAbsoluteDeviation(BaseAnomalyModel):

  def __init__(self, window_size=10, scale_factor=0.67449):
    self._median_tracker = univariate.SimpleMedianTracker(window_size)
    self._mad_tracker = univariate.SimpleMADTracker(window_size)
    self._scale_factor = scale_factor

  def learn_one(self, x: beam.Row):
    assert len(x.__dict__) == 1, "MAD requires univariate input"
    v = next(iter(x))
    self._median_tracker.push(v)
    self._mad_tracker.push(v)

  def score_one(self, x: beam.Row):
    assert len(x.__dict__) == 1, "MAD requires univariate input"
    v = next(iter(x))
    median = self._median_tracker.get()
    mad = self._mad_tracker.get()
    if mad < EPSILON:
      return float('NaN')
    return abs((v - median) / mad * self._scale_factor)