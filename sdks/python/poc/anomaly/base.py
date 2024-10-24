"""Base classes for anomaly detection"""

# from typing import Iterable
import math
from typing import Optional
from typing import TypeVar
from typing import List
import uuid

import apache_beam as beam
from apache_beam.utils import timestamp
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec
from apache_beam.coders import DillCoder

import numpy as np
import river
import river.anomaly

from poc.anomaly import univariate

InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')

__all__ = ['AnomalyDetector', 'EnsembleAnomalyDetector', 'AnomalyDetection']


class StandardAbsoluteDeviation(river.anomaly.base.AnomalyDetector):

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

  def learn_one(self, x: dict):
    assert len(x) == 1, "SAD requires univariate input"
    k = next(iter(x))
    v = x[k]
    self._stdev_tracker.push(v)
    self._sub_stat_tracker.push(v)

  def score_one(self, x: dict):
    assert len(x) == 1, "SAD requires univariate input"
    k = next(iter(x))
    v = x[k]
    sub_stat = self._sub_stat_tracker.get()
    stdev = self._stdev_tracker.get()
    if math.isnan(stdev) or abs(stdev) < 1e-12:
      return 0.0
    return abs((v - sub_stat) / stdev)


class MedianAbsoluteDeviation(river.anomaly.base.AnomalyDetector):

  def __init__(self, window_size=10, scale_factor=0.67449):
    self._median_tracker = univariate.SimpleMedianTracker(window_size)
    self._mad_tracker = univariate.SimpleMADTracker(window_size)
    self._scale_factor = scale_factor

  def learn_one(self, x: dict):
    assert len(x) == 1, "MAD requires univariate input"
    k = next(iter(x))
    v = x[k]
    self._median_tracker.push(v)
    self._mad_tracker.push(v)

  def score_one(self, x: dict):
    assert len(x) == 1, "MAD requires univariate input"
    k = next(iter(x))
    v = x[k]
    median = self._median_tracker.get()
    mad = self._mad_tracker.get()
    return abs((v - median) / mad * self._scale_factor)


KNOWN_SUPERVISED_ALGORITHMS = {}

KNOWN_UNSUPERVISED_ALGORITHMS = {
    "SAD": StandardAbsoluteDeviation,
    "MAD": MedianAbsoluteDeviation,
    "iLOF": river.anomaly.lof.LocalOutlierFactor,
    # "HSF": river.anomaly.HalfSpaceTrees,
    # "OneClassSVM": river.anomaly.OneClassSVM,
}

KNOWN_ALGORITHMS = KNOWN_SUPERVISED_ALGORITHMS | KNOWN_UNSUPERVISED_ALGORITHMS


class AnomalyDetector:

  def __init__(self,
               algorithm: Optional[str] = None,
               label: Optional[str] = None,
               fields=None,
               target=None,
               *args,
               **kwargs) -> None:
    if algorithm in KNOWN_ALGORITHMS:
      detector = KNOWN_ALGORITHMS[algorithm]
      self._underlying = detector(*args, **kwargs)
      if label:
        self._label = label
      else:
        self._label = f"{algorithm}_{uuid.uuid4().hex}"
      if algorithm in KNOWN_SUPERVISED_ALGORITHMS:
        self._is_supervised = True
      else:
        self._is_supervised = False
    else:
      raise NotImplementedError

    self._fields = fields
    self._target = target

  @property
  def label(self):
    return self._label

  @property
  def is_supervised(self):
    return self._is_supervised

  def score_and_learn(self, x, y, unused_key):
    y_pred = self._underlying.score_one(x)
    self._underlying.learn_one(x)
    return y_pred


class LodaWeakLearner(AnomalyDetector):

  def __init__(self, fields, target):
    self._fields = fields
    self._target = target
    n_fields = len(fields)
    self._projection = np.random.randn(n_fields)
    n_nonzero_dims = int(np.sqrt(n_fields))
    zero_idx = np.random.permutation(len(fields))[:(n_fields - n_nonzero_dims)]
    self._projection[zero_idx] = 0
    self._hist = univariate.SimpleHistogram(window_size=256, n_bins=256)
    self._label = f"LodaWeakLearner_{uuid.uuid4().hex}"

  def score_and_learn(self, x, y, unused_key):
    x_np = np.array([x[k] for k in self._fields])
    projected_data = x_np.dot(self._projection)

    if (len(self._hist._queue) < 256):
      y_pred = 0
    else:
      histogram, limits = self._hist.get()
      histogram = histogram.astype(np.float64)
      histogram += 1e-12
      histogram /= np.sum(histogram)

      inds = np.searchsorted(limits[:256 - 1], projected_data, side='left')
      y_pred = -np.log(histogram[inds])

    self._hist.push(projected_data)

    return y_pred


class EnsembleAnomalyDetector(AnomalyDetector):

  def __init__(self, n, weak_learner, aggregator=None, *args, **kvargs):
    self._weak_learners = []
    self._label = f"Ensemble-{weak_learner}_{uuid.uuid4().hex}"
    for _ in range(n):
      if weak_learner == "loda":
        self._weak_learners.append(LodaWeakLearner(*args, **kvargs))
    self._agg = aggregator

  def score_and_learn(self, x, y, unused_key):
    raise NotImplementedError()


class ScoreAndLearn(beam.DoFn):
  DETECTOR_STATE_INDEX = ReadModifyWriteStateSpec('saved_detector', DillCoder())

  def __init__(self, detector):
    self.detector = detector

  def process(self,
              element,
              detector_state=beam.DoFn.StateParam(DETECTOR_STATE_INDEX),
              **kwargs):

    _, (k, v) = element
    detector = detector_state.read() # type: ignore
    if detector is None:
      detector = self.detector

    # field selection
    if detector._fields is not None:
      x = {field: getattr(v, field) for field in detector._fields}
    else:
      x = v._asdict

    if detector._target is not None:
      y = getattr(v, detector._target)
    else:
      y = None

    yield k, (v, beam.Row(model=detector.label, score=detector.score_and_learn(x, y, k)))

    detector_state.write(detector) # type: ignore


class EvaluateWithAUC(beam.DoFn):
  TRACKER_STATE_INDEX = ReadModifyWriteStateSpec('saved_trackers', DillCoder())

  def __init__(self, window_size, target):
    self._window_size = window_size
    self._target = target

  def process(self,
              element,
              tracker_state=beam.DoFn.StateParam(TRACKER_STATE_INDEX),
              **kwargs):

    _, elem = element
    trackers = tracker_state.read() # type: ignore
    if trackers is None:
      trackers = {}

    key, (data, result) = elem
    target = getattr(data, self._target)

    if result.model not in trackers:
      from river.metrics import RollingROCAUC
      trackers[result.model] = RollingROCAUC(
          window_size=self._window_size)
    trackers[result.model].update(target, result.score)

    auc = trackers[result.model].get()
    yield key, (data, beam.Row(auc=auc, **result._asdict()))
    tracker_state.write(trackers) # type: ignore


class AverageFn(beam.CombineFn):

  def create_accumulator(self):
    sum = 0.0
    count = 0
    accumulator = sum, count
    return accumulator

  def add_input(self, accumulator, input):
    sum, count = accumulator
    return sum + input, count + 1

  def merge_accumulators(self, accumulators):
    sums, counts = zip(*accumulators)
    return sum(sums), sum(counts)

  def extract_output(self, accumulator):
    sum, count = accumulator
    if count == 0:
      return float('NaN')
    return sum / count


class AnomalyDetection(beam.PTransform[beam.PCollection[InputT],
                                       beam.PCollection[OutputT]]):

  def __init__(self,
               detectors: Optional[List[AnomalyDetector]] = None,
               agg_strategy=None,
               is_nested=False,
               with_auc=False) -> None:
    self._detectors = detectors
    self._with_auc = with_auc
    self._is_nested = is_nested

  def maybe_add_key(self, element):
    # TODO: may not need to add keys if there is an existing one?
    key = timestamp.Timestamp.now().micros
    return key, element

  def expand(self,
             input_row: beam.PCollection[InputT]) -> beam.PCollection[OutputT]:

    assert self._detectors is not None

    if not self._is_nested:
      data = (
          input_row | "Add temp key" >> beam.Map(self.maybe_add_key)
          | "Add dummy key for stateful doFn" >> beam.WithKeys(1))
    else:
      data = input_row

    model_results = []
    for detector in self._detectors:
      if isinstance(detector, EnsembleAnomalyDetector):
        # call AnomalyDetection PTransform recursively for ensemble
        model_results.append(
            data
            | f"Score and learn ({detector})" >> AnomalyDetection(
                detector._weak_learners, is_nested=True)
            # x is (temp_key, (data, Row(model, score)))
            | f"Pre-combine map ({detector})" >> beam.Map(lambda x: (
                (x[0], (x[1][0])), x[1][1].score))
            | f"CombinePerKey ({detector}" >> beam.CombinePerKey(AverageFn())
            | f"Post-combine map ({detector})" >> beam.Map(
                lambda x, label=detector.label: (x[0][0],
                                                 (x[0][1],
                                                  beam.Row(model=label,
                                                           score=x[1])))))
      else:
        model_results.append(data
                             | f"Score and learn ({detector})" >> beam.ParDo(
                                 ScoreAndLearn(detector)))

    # (temp_key, (data, Row(model, score)))
    merged = model_results | beam.Flatten()
    ret = merged

    if not self._is_nested:
      if self._with_auc:
        ret = (
            merged
            | "Add dummy key again" >> beam.WithKeys(1)
            | "EvaluateAUC" >> beam.ParDo(EvaluateWithAUC(1000, "label")))

      # remove temp_key
      ret = ret | beam.Values()

    return ret # type: ignore
