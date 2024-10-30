
import logging
from typing import Optional

import apache_beam as beam
from apache_beam.ml.anomaly.base import BaseAnomalyModel
from apache_beam.ml.anomaly.models.sad import StandardAbsoluteDeviation
from apache_beam.ml.anomaly.models.mad import MedianAbsoluteDeviation
from apache_beam.ml.anomaly.models.loda import LodaWeakLearner

try:
  from apache_beam.ml.anomaly.models.river import LocalOutlierFactor # type: ignore
except ImportError:
  logging.warning("Unable to import river model 'LocalOutlierFactor'")
  LocalOutlierFactor = None


class DummyEnsembleAnomalyModel(BaseAnomalyModel):
  def __init__(*args, **kwargs):
    pass

  def learn_one(self, x: beam.Row) -> None:
    raise NotImplementedError("This function should not be called.")

  def score_one(self, x: beam.Row) -> float:
    raise NotImplementedError("This function should not be called.")


KNOWN_ALGORITHMS: dict[str, Optional[type[BaseAnomalyModel]]] = {
    "sad": StandardAbsoluteDeviation,
    "mad": MedianAbsoluteDeviation,
    "loda": LodaWeakLearner,
    "ilof": LocalOutlierFactor,
    "ensemble": DummyEnsembleAnomalyModel,  # a wrapper for ensemble algorithms
}

# Remove unavailable algorithms
KNOWN_ALGORITHMS = {k: v for k, v in KNOWN_ALGORITHMS.items() if v is not None}
