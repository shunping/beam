from typing import Any
from typing import Iterable
from typing import Tuple
from typing import Union

import apache_beam as beam
from apache_beam.coders import DillCoder
from apache_beam.ml.anomaly import univariate
from apache_beam.ml.anomaly.base import AnomalyPrediction
from apache_beam.ml.anomaly.base import BaseThresholdFunc
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec


class FixedThreshold(BaseThresholdFunc):

  def __init__(self, threshold: Union[int, float]):
    self._threshold = threshold

  @property
  def threshold(self):
    return self._threshold

  def process(self, element: Tuple[Any, AnomalyPrediction],
              **kwargs) -> Iterable[Tuple[Any, AnomalyPrediction]]:
    key, prediction = element
    yield key, self._update_prediction(prediction)


class QuantileThreshold(BaseThresholdFunc):
  TRACKER_STATE_INDEX = ReadModifyWriteStateSpec('saved_tracker', DillCoder())

  def __init__(self, quantile: float):
    self._quantile = quantile
    self._tracker = None

  @property
  def threshold(self) -> float:
    return self._tracker.get()  # type: ignore

  def process(self,
              element: Tuple[Any, Tuple[Any, AnomalyPrediction]],
              tracker_state=beam.DoFn.StateParam(TRACKER_STATE_INDEX),
              **kwargs) -> Iterable[Tuple[Tuple[Any, Any], AnomalyPrediction]]:
    key1, (key2, prediction) = element

    self._tracker = tracker_state.read()  # type: ignore
    if self._tracker is None:
      self._tracker = univariate.SimpleQuantile(100, self._quantile)
    self._tracker.push(prediction.decision.score)

    yield (key1, key2), self._update_prediction(prediction)

    tracker_state.write(self._tracker)  # type: ignore
