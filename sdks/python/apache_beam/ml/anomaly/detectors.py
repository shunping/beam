
from typing import Optional
from typing import List
import uuid

from apache_beam.ml.anomaly.base import AggregationStrategy
from apache_beam.ml.anomaly.base import BaseThresholdFunc
from apache_beam.ml.anomaly.aggregations import average_score

from apache_beam.ml.anomaly.models import KNOWN_ALGORITHMS


class AnomalyDetector:

  def __init__(self,
               algorithm: str,
               id: Optional[str] = None,
               features: Optional[List[str]] = None,
               target: Optional[str] = None,
               threshold_func: Optional[BaseThresholdFunc] = None,
               *args,
               **kwargs) -> None:
    algorithm = algorithm.lower()
    if algorithm in KNOWN_ALGORITHMS:
      detector = KNOWN_ALGORITHMS[algorithm]
      if detector is not None:
        self._underlying = detector(*args, **kwargs)
    else:
      raise NotImplementedError(f"algorithm '{algorithm}' not found")

    self._id = id if id else f"{algorithm}_{uuid.uuid4().hex[:6]}"
    self._features = features
    self._target = target
    self._threshold_func = threshold_func

  @property
  def label(self):
    return self._id

  def __repr__(self):
    return self.label

  def score_and_learn(self, x, y, unused_key):
    y_pred = self._underlying.score_one(x)
    self._underlying.learn_one(x)
    return y_pred


class EnsembleAnomalyDetector(AnomalyDetector):

  def __init__(
      self,
      n: int = 10,
      label: Optional[str] = None,
      aggregation_strategy: Optional[AggregationStrategy] = average_score(),
      **kwargs):
    weak_learner_alg = kwargs["algorithm"]
    kwargs["algorithm"] = "ensemble"
    super().__init__(label=label, **kwargs)

    kwargs["algorithm"] = weak_learner_alg
    self._weak_learners = []
    for _ in range(n):
      self._weak_learners.append(AnomalyDetector(**kwargs))

    self._aggregation_strategy = aggregation_strategy

  def score_and_learn(self, x, y, unused_key):
    raise NotImplementedError()
