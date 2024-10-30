import dataclasses
import logging
from typing import Any
from typing import Callable
from typing import Generic
from typing import Iterable
from typing import Tuple
from typing import Optional
from typing import TypeVar
from typing import Union

import apache_beam as beam
from apache_beam.coders import DillCoder
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec
from apache_beam.utils import timestamp
from apache_beam.runners.common import DoFnSignature

from apache_beam.ml.anomaly.base import AggregationStrategy
from apache_beam.ml.anomaly.base import AnomalyDecision
from apache_beam.ml.anomaly.base import AnomalyPrediction
from apache_beam.ml.anomaly.detectors import AnomalyDetector
from apache_beam.ml.anomaly.detectors import EnsembleAnomalyDetector

TempKey = TypeVar('TempKey', int, Any)


class ScoreAndLearn(beam.DoFn, Generic[TempKey]):
  DETECTOR_STATE_INDEX = ReadModifyWriteStateSpec('saved_detector', DillCoder())

  def __init__(self, detector):
    self.detector = detector

  def process(
      self,
      element: Tuple[Any, Tuple[TempKey, beam.Row]],
      detector_state=beam.DoFn.StateParam(DETECTOR_STATE_INDEX),
      **kwargs) -> Iterable[Tuple[Tuple[Any, TempKey], AnomalyPrediction]]:

    k1, (k2, data) = element
    detector = detector_state.read()  # type: ignore
    if detector is None:
      detector = self.detector

    # feature selection
    if detector._features is not None:
      x = beam.Row(**{f: getattr(data, f) for f in detector._features})
    else:
      x = data

    if detector._target is not None:
      y = getattr(data, detector._target)
    else:
      y = None

    yield (k1, k2), AnomalyPrediction(
        data=data,
        decision=AnomalyDecision(
            model=detector.label, score=detector.score_and_learn(x, y, k2)))

    detector_state.write(detector)  # type: ignore


class EvaluateWithAUC(beam.DoFn):
  TRACKER_STATE_INDEX = ReadModifyWriteStateSpec('saved_trackers', DillCoder())

  def __init__(self, window_size, target):
    self._window_size = window_size
    self._target = target

  def process(self,
              element: Tuple[Any, AnomalyPrediction],
              tracker_state=beam.DoFn.StateParam(TRACKER_STATE_INDEX),
              **kwargs) -> Iterable[Tuple[Any, AnomalyPrediction]]:

    key, prediction = element
    data = prediction.data
    decision = prediction.decision

    if not self._target:
      auc = float("nan")
    else:
      target = getattr(data, self._target)

      trackers = tracker_state.read()  # type: ignore
      if trackers is None:
        trackers = {}

      if decision.model not in trackers:
        from river.metrics import RollingROCAUC
        trackers[decision.model] = RollingROCAUC(window_size=self._window_size)
      trackers[decision.model].update(target, decision.score)

      auc = trackers[decision.model].get()

    yield key, AnomalyPrediction(
        data=data, decision=dataclasses.replace(decision, auc=auc))
    tracker_state.write(trackers)  # type: ignore


class AnomalyDetection(
    beam.PTransform[beam.PCollection[Tuple[Any, beam.Row]],
                    beam.PCollection[Tuple[Any, AnomalyPrediction]]],
    Generic[TempKey]):

  def __init__(self,
               detectors: Iterable[AnomalyDetector],
               aggregation_strategy: Optional[AggregationStrategy] = None,
               is_nested: bool = False,
               with_auc: bool = False) -> None:
    self._detectors = detectors
    self._with_auc = with_auc
    self._is_nested = is_nested
    self._aggregation_strategy = aggregation_strategy

  def maybe_add_key(
      self, element: Tuple[Any,
                           beam.Row]) -> Tuple[Any, Tuple[TempKey, beam.Row]]:
    key, row = element
    return key, (timestamp.Timestamp.now().micros, row)  # type: ignore

  def expand(
      self, input: Union[beam.PCollection[Tuple[Any, beam.Row]],
                         beam.PCollection[Tuple[Any, Tuple[TempKey, beam.Row]]]]
  ) -> beam.PCollection[Tuple[Any, AnomalyPrediction]]:

    assert self._detectors is not None

    if self._is_nested:
      data = input
    else:
      data = (input | "Add temp key" >> beam.Map(self.maybe_add_key))

    model_results = []
    for detector in self._detectors:
      if isinstance(detector, EnsembleAnomalyDetector):
        # call AnomalyDetection PTransform recursively for ensemble
        score_result = (
            data
            | f"Score and learn ({detector})" >> AnomalyDetection(
                detector._weak_learners,
                aggregation_strategy=detector._aggregation_strategy,
                is_nested=True)
            | f"Reset model label for ensemble ({detector})" >>
            beam.MapTuple(lambda k, v, label=detector.label: (
                (k[0], k[1]),
                AnomalyPrediction(
                    data=v.data,
                    decision=dataclasses.replace(v.decision, model=label))))
            .with_output_types(Tuple[Tuple[Any, TempKey], AnomalyPrediction]))
      else:
        score_result = (
            data
            | f"Score and learn ({detector})" >> beam.ParDo(
                ScoreAndLearn(detector)).with_output_types(
                    Tuple[Tuple[Any, TempKey], AnomalyPrediction]))

      if detector._threshold_func:
        if DoFnSignature(detector._threshold_func).is_stateful_dofn():
          model_results.append(
              score_result
              | f"Re-arrange keys for stateful threshold function ({detector})"
              >> beam.MapTuple(lambda k, v: (k[0], (k[1], v)))
              | f"Run threshold function ({detector})" >> beam.ParDo(
                  detector._threshold_func))
        else:
          model_results.append(score_result
                               | f"Run threshold function ({detector})" >>
                               beam.ParDo(detector._threshold_func))
      else:
        model_results.append(score_result)

    # (key, temp_key), AnomalyPrediction
    merged = model_results | beam.Flatten()

    ret = merged

    if self._aggregation_strategy is not None:
      ret = (
          ret | beam.MapTuple(lambda k, v: ((k[0], k[1], v.data), v.decision))
          | beam.GroupByKey()
          | beam.MapTuple(lambda k, v, agg=self._aggregation_strategy: (
              (k[0], k[1]), AnomalyPrediction(data=k[2], decision=agg(v)))))

    if not self._is_nested:
      remove_temp_key_func: Callable[
          [Tuple[Any, TempKey], AnomalyPrediction],
          Tuple[Any, AnomalyPrediction]] = lambda k, v: (k[0], v)
      ret |= beam.MapTuple(remove_temp_key_func)

      if self._with_auc:
        target = None
        for detector in self._detectors:
          if detector._target is None:
            logging.warning(
                "anomaly detector %s does not specify target for auc", detector)
          elif target is None:
            target = detector._target
            logging.info("compute auc with target {target}")
          elif target != detector._target:
            logging.warning(
                "anomaly detector %s has an unmatched target for auc", detector)

        ret |= "EvaluateAUC" >> beam.ParDo(EvaluateWithAUC(1000, target))

    return ret  # type: ignore
