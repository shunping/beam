"""Base classes for anomaly detection"""

import dataclasses
from dataclasses import dataclass
from typing import Optional
from typing import Protocol
from typing import Iterable
from typing import Union

import apache_beam as beam


@dataclass(frozen=True)
class AnomalyDecision():
  model: str = ""
  score: float = float('NaN')
  auc: Optional[float] = None
  prediction: Optional[int] = None
  threshold: Optional[float] = None
  info: str = ''


@dataclass(frozen=True)
class AnomalyPrediction():
  data: beam.Row
  decision: AnomalyDecision


class BaseThresholdFunc(beam.DoFn):

  @property
  def threshold(self) -> Union[int, float]:
    raise NotImplementedError

  def _update_prediction(self,
                         prediction: AnomalyPrediction) -> AnomalyPrediction:
    pred: int = 0 if prediction.decision.score < self.threshold else 1  # type: ignore
    return dataclasses.replace(
        prediction,
        decision=dataclasses.replace(
            prediction.decision, prediction=pred, threshold=self.threshold))


class AggregationStrategy(Protocol):
  def __call__(self, decisions:Iterable[AnomalyDecision]) -> AnomalyDecision:
    ...
