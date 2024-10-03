from abc import ABC, abstractmethod
import math
from typing import Optional
from typing import List
import uuid

import numpy as np
import river
import river.anomaly

import apache_beam as beam
from poc.anomaly import univariate
from poc.anomaly.base import AggregationStrategy
from poc.anomaly.base import BaseThresholdFunc
from poc.anomaly.aggregations import average_score


class BaseAnomalyModel(ABC):

  @abstractmethod
  def learn_one(self, x: beam.Row) -> None:
    pass

  @abstractmethod
  def score_one(self, x: beam.Row) -> float:
    pass


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
    if math.isnan(stdev) or abs(stdev) < 1e-12:
      return 0.0
    return abs((v - sub_stat) / stdev)


EPS = 1e-12


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
    if mad < EPS:
      return float('NaN')
    return abs((v - median) / mad * self._scale_factor)


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


KNOWN_ALGORITHMS: dict[str, Optional[type[BaseAnomalyModel]]] = {
    "sad": StandardAbsoluteDeviation,
    "mad": MedianAbsoluteDeviation,
    "loda": LodaWeakLearner,
    "ilof": LocalOutlierFactor,
    # # "HSF": river.anomaly.HalfSpaceTrees,
    # # "OneClassSVM": river.anomaly.OneClassSVM,
    "ensemble": None,  # umbrella for all ensemble algorithms
}


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
