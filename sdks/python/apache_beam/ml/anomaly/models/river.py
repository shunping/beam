import river.anomaly

import apache_beam as beam
from apache_beam.ml.anomaly.base import BaseAnomalyModel

class RiverAnomalyModel(BaseAnomalyModel):
  def __init__(self):
    self._river_model = None

  def learn_one(self, x: beam.Row):
    self._river_model.learn_one(x.__dict__) # type: ignore

  def score_one(self, x: beam.Row):
    return self._river_model.score_one(x.__dict__) # type: ignore

class LocalOutlierFactor(RiverAnomalyModel):
  def __init__(self, *args, **kwargs):
    self._river_model = river.anomaly.LocalOutlierFactor(*args, **kwargs)