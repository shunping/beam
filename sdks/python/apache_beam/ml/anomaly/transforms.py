#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import dataclasses
from typing import Callable
from typing import Generic
from typing import Iterable
from typing import Tuple
from typing import Optional
from typing import TypeVar

import apache_beam as beam
from apache_beam.coders import DillCoder
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec
from apache_beam.utils import timestamp

from apache_beam.ml.anomaly.base import AnomalyPrediction
from apache_beam.ml.anomaly.base import AnomalyResult
from apache_beam.ml.anomaly.base import BaseAggregationFunc
from apache_beam.ml.anomaly.detectors import AnomalyDetector
from apache_beam.ml.anomaly.detectors import EnsembleAnomalyDetector
from apache_beam.ml.anomaly.models import KNOWN_ALGORITHMS

KeyT = TypeVar('KeyT')
TempKeyT = TypeVar('TempKeyT')


class _ScoreAndLearn(beam.DoFn):
  DETECTOR_STATE_INDEX = ReadModifyWriteStateSpec('saved_detector', DillCoder())

  def __init__(self, detector: AnomalyDetector):
    self._detector = detector
    self._canonical_alg = self._detector.algorithm.lower()
    if not self._canonical_alg in KNOWN_ALGORITHMS:
      raise NotImplementedError(f"algorithm '{detector.algorithm}' not found")

  def score_and_learn(self, x, y, unused_key):
    y_pred = self._underlying.score_one(x)
    self._underlying.learn_one(x)
    return y_pred

  def process(
      self,
      element: Tuple[KeyT, Tuple[TempKeyT, beam.Row]],
      detector_state=beam.DoFn.StateParam(DETECTOR_STATE_INDEX),
      **kwargs) -> Iterable[Tuple[KeyT, Tuple[TempKeyT, AnomalyResult]]]:

    k1, (k2, data) = element
    self._underlying = detector_state.read()  # type: ignore
    if self._underlying is None:
      model_class = KNOWN_ALGORITHMS[self._canonical_alg]
      kwargs = self._detector.algorithm_kwargs if self._detector.algorithm_kwargs is not None else {}
      self._underlying = model_class(**kwargs)  # type: ignore

    # TODO: can we get rid of this conversion?
    # TODO: handle missing fields
    # feature selection
    if self._detector.features is not None:
      x = beam.Row(**{f: getattr(data, f) for f in self._detector.features})
    else:
      x = beam.Row(**data._asdict())

    if self._detector.target is not None:
      y = getattr(data, self._detector.target)
    else:
      y = None

    yield k1, (k2,
               AnomalyResult(
                   example=data,
                   prediction=AnomalyPrediction(
                       model_id=self._detector.id,
                       score=self.score_and_learn(x, y, k2))))

    detector_state.write(self._underlying)  # type: ignore


class _EvaluateWithAUC(beam.DoFn):
  TRACKER_STATE_INDEX = ReadModifyWriteStateSpec('saved_trackers', DillCoder())

  def __init__(self, window_size, target):
    self._window_size = window_size
    self._target = target

  def process(self,
              element: Tuple[KeyT, AnomalyResult],
              tracker_state=beam.DoFn.StateParam(TRACKER_STATE_INDEX),
              **kwargs) -> Iterable[Tuple[KeyT, AnomalyResult]]:

    key, prediction = element
    data = prediction.example
    decision = prediction.prediction

    if not self._target:
      auc = float("nan")
    else:
      target = getattr(data, self._target)

      trackers = tracker_state.read()  # type: ignore
      if trackers is None:
        trackers = {}

      if decision.model_id not in trackers:
        from river.metrics import RollingROCAUC
        trackers[decision.model_id] = RollingROCAUC(window_size=self._window_size)
      trackers[decision.model_id].update(target, decision.score)

      auc = trackers[decision.model_id].get()

    yield key, AnomalyResult(
        example=data, prediction=dataclasses.replace(decision, auc=auc))
    tracker_state.write(trackers)  # type: ignore


class _RunDetectors(
    beam.PTransform[beam.PCollection[Tuple[KeyT, Tuple[TempKeyT, beam.Row]]],
                    beam.PCollection[Tuple[KeyT, Tuple[TempKeyT,
                                                       AnomalyResult]]]],
    Generic[KeyT, TempKeyT]):

  def __init__(self,
               model_id,
               detectors: Iterable[AnomalyDetector],
               aggregation_strategy: Optional[BaseAggregationFunc] = None):
    self._label = model_id
    self._detectors = detectors
    self._aggregation_strategy = aggregation_strategy

  def expand(
      self, input: beam.PCollection[Tuple[KeyT, Tuple[TempKeyT, beam.Row]]]
  ) -> beam.PCollection[Tuple[KeyT, Tuple[TempKeyT, AnomalyResult]]]:
    model_results = []
    for detector in self._detectors:
      if isinstance(detector, EnsembleAnomalyDetector):
        score_result = (
            input | _RunDetectors(
                detector.id,
                detector.weak_learners,  # type: ignore
                detector.aggregation_strategy)
            | f"Reset model label for ensemble ({detector})" >>
            beam.MapTuple(lambda k, v, label=detector.id: (k, (
                v[0],
                AnomalyResult(
                    example=v[1].example,
                    prediction=dataclasses.replace(v[1].prediction, model_id=label)))))
            .with_output_types(Tuple[KeyT, Tuple[TempKeyT, AnomalyResult]]))
      else:
        score_result = (
            input
            | f"Reshuffle ({detector})" >> beam.Reshuffle()
            | f"Score and learn ({detector})" >> beam.ParDo(
                _ScoreAndLearn(detector)).with_output_types(
                    Tuple[KeyT, Tuple[TempKeyT, AnomalyResult]]))

      if detector.threshold_func:
        model_results.append(score_result
                             | f"Run threshold function ({detector})" >>
                             beam.ParDo(detector.threshold_func))
      else:
        model_results.append(score_result)

    merged = model_results | f"Flatten {self._label}" >> beam.Flatten()

    ret = merged
    if self._aggregation_strategy is not None:
      ret = (
          ret
          | beam.MapTuple(lambda k, v: ((k, v[0]), v[1]))
          | beam.GroupByKey()
          | beam.MapTuple(lambda k, v, agg=self._aggregation_strategy: (k[0], (
              k[1],
              AnomalyResult(
                  example=v[0].example,
                  prediction=agg([result.prediction for result in v])))))
          .with_output_types(Tuple[KeyT, Tuple[TempKeyT, AnomalyResult]]))

    return ret


class AnomalyDetection(
    beam.PTransform[beam.PCollection[Tuple[KeyT, beam.Row]],
                    beam.PCollection[Tuple[KeyT, AnomalyResult]]],
    Generic[KeyT, TempKeyT]):

  def __init__(self,
               detectors: Iterable[AnomalyDetector],
               aggregation_func: Optional[BaseAggregationFunc] = None,
               is_nested: bool = False,
               with_auc: bool = False) -> None:
    self._detectors = detectors
    self._with_auc = with_auc
    self._is_nested = is_nested
    self._aggregation_func = aggregation_func

  def maybe_add_key(
      self, element: Tuple[KeyT,
                           beam.Row]) -> Tuple[KeyT, Tuple[TempKeyT, beam.Row]]:
    key, row = element
    return key, (timestamp.Timestamp.now().micros, row)  # type: ignore

  def expand(
      self,
      input: beam.PCollection[Tuple[KeyT, beam.Row]],
  ) -> beam.PCollection[Tuple[KeyT, AnomalyResult]]:

    assert self._detectors is not None

    ret = (
        input
        | "Add temp key" >> beam.Map(self.maybe_add_key)
        | _RunDetectors("root", self._detectors, self._aggregation_func))

    remove_temp_key_func: Callable[
        [KeyT, Tuple[TempKeyT, AnomalyResult]],
        Tuple[KeyT, AnomalyResult]] = lambda k, v: (k, v[1])
    ret |= beam.MapTuple(remove_temp_key_func).with_input_types(
        Tuple[KeyT, Tuple[TempKeyT, AnomalyResult]]).with_output_types(
            Tuple[KeyT, AnomalyResult])

    return ret  # type: ignore
