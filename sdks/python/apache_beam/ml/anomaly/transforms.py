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

from typing import Any
from typing import Callable
from typing import cast
from typing import Iterable
from typing import List
from typing import Tuple
from typing import Optional
import uuid

import apache_beam as beam
from apache_beam.coders import DillCoder
from apache_beam.ml.anomaly import thresholds
from apache_beam.ml.anomaly.base import AnomalyDetector
from apache_beam.ml.anomaly.base import AnomalyPrediction
from apache_beam.ml.anomaly.base import AnomalyResult
from apache_beam.ml.anomaly.base import AggregationFn
from apache_beam.ml.anomaly.base import ThresholdFn
from apache_beam.ml.anomaly.base import EnsembleAnomalyDetector
from apache_beam.ml.anomaly.configurable import Config
from apache_beam.ml.anomaly.configurable import Configurable
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec
from apache_beam.transforms.userstate import ReadModifyWriteRuntimeState
from apache_beam.utils import timestamp


class _ScoreAndLearn(beam.DoFn):
  MODEL_STATE_INDEX = ReadModifyWriteStateSpec('saved_model', DillCoder())

  def __init__(self, detector_config: Config):
    self._detector_config = detector_config
    self._detector_config.args["initialize_model"] = True
    # object.__delattr__(self._detector, "algorithm_args")
    # self._canonical_alg = self._detector.type.lower()
    # if not self._canonical_alg in KNOWN_ALGORITHMS:
    #   raise NotImplementedError(f"algorithm '{detector.type}' not found")

  def score_and_learn(self, data):
    assert self._underlying
    if self._underlying._features is not None:
      x = beam.Row(**{f: getattr(data, f) for f in self._underlying._features})
    else:
      x = beam.Row(**data._asdict())
    y_pred = self._underlying.score_one(x)
    self._underlying.learn_one(x)
    return y_pred

  def process(self,
              element: Tuple[Any, Tuple[Any, beam.Row]],
              model_state=beam.DoFn.StateParam(MODEL_STATE_INDEX),
              **kwargs) -> Iterable[Tuple[Any, Tuple[Any, AnomalyResult]]]:

    model_state = cast(ReadModifyWriteRuntimeState, model_state)
    k1, (k2, data) = element
    self._underlying: AnomalyDetector = model_state.read()
    if self._underlying is None:
      self._underlying = cast(AnomalyDetector,
                              Configurable.from_config(self._detector_config))

    yield k1, (k2,
               AnomalyResult(
                   example=data,
                   prediction=AnomalyPrediction(
                       model_id=self._underlying._model_id,
                       score=self.score_and_learn(data))))

    model_state.write(self._underlying)


class _RunThresholdCriterion(
    beam.PTransform[beam.PCollection[Tuple[Any, Tuple[Any, beam.Row]]],
                    beam.PCollection[Tuple[Any, Tuple[Any, AnomalyResult]]]]):

  def __init__(self, model_id, threshold_criterion):
    self._model_id = model_id
    self._threshold_criterion = threshold_criterion

  def expand(
      self, input: beam.PCollection[Tuple[Any, Tuple[Any, AnomalyResult]]]
  ) -> beam.PCollection[Tuple[Any, Tuple[Any, AnomalyResult]]]:

    # threshold_fn = ThresholdFn.from_config(self._threshold_criterion)
    threshold_fn = self._threshold_criterion
    if threshold_fn:
      if threshold_fn.is_stateful:
        postprocess_result = (
            input
            | beam.ParDo(
                thresholds.StatefulThresholdDoFn(threshold_fn.to_config())))
      else:
        postprocess_result = (
            input
            | beam.ParDo(
                thresholds.StatelessThresholdDoFn(threshold_fn.to_config())))
    else:
      postprocess_result: Any = input

    return postprocess_result


class _RunOneDetector(
    beam.PTransform[beam.PCollection[Tuple[Any, Tuple[Any, beam.Row]]],
                    beam.PCollection[Tuple[Any, Tuple[Any, AnomalyResult]]]]):

  def __init__(self, detector):
    # self._detector = AnomalyDetector.from_config(detector)
    self._detector = detector

  def expand(
      self, input: beam.PCollection[Tuple[Any, Tuple[Any, beam.Row]]]
  ) -> beam.PCollection[Tuple[Any, Tuple[Any, AnomalyResult]]]:
    model_uuid = f"{self._detector._model_id}:{uuid.uuid4().hex[:6]}"
    result: Any = (
        input
        | beam.Reshuffle()
        | f"Score and Learn ({model_uuid})" >> beam.ParDo(
            _ScoreAndLearn(self._detector.to_config())).with_output_types(
                Tuple[Any, Tuple[Any, AnomalyResult]])
        | f"Run Threshold Criterion ({model_uuid})" >> _RunThresholdCriterion(
            self._detector._model_id, self._detector._threshold_criterion))

    return result


class _RunEnsembleDetector(
    beam.PTransform[beam.PCollection[Tuple[Any, Tuple[Any, beam.Row]]],
                    beam.PCollection[Tuple[Any, Tuple[Any, AnomalyResult]]]]):

  def __init__(self, ensemble_detector: EnsembleAnomalyDetector):
    self._ensemble_detector = ensemble_detector

  def expand(
      self, input: beam.PCollection[Tuple[Any, Tuple[Any, beam.Row]]]
  ) -> beam.PCollection[Tuple[Any, Tuple[Any, AnomalyResult]]]:
    model_uuid = f"{self._ensemble_detector._model_id}:{uuid.uuid4().hex[:6]}"

    model_results = []
    assert self._ensemble_detector._learners is not None
    if not self._ensemble_detector._learners:
      raise ValueError(f"No detectors found at {model_uuid}")

    for idx, detector in enumerate(self._ensemble_detector._learners):
      if isinstance(detector, EnsembleAnomalyDetector):
        score_result = (
            input | f"Run Ensemble Detector at index {idx} ({model_uuid})" >>
            _RunEnsembleDetector(detector))
      else:
        score_result = (
            input
            | f"Run One Detector at index {idx} ({model_uuid})" >>
            _RunOneDetector(detector))
      model_results.append(score_result)

    merged = (model_results | beam.Flatten())

    ret: Any = merged
    # aggregation_strategy = AggregationFn.from_config(
    #     self._ensemble_detector._aggregation_strategy)
    aggregation_strategy = self._ensemble_detector._aggregation_strategy

    if aggregation_strategy is not None:
      # if no model_override is set in the aggregation function, use
      # model id locally in the instance

      if getattr(aggregation_strategy, "_model_override") is None:
        setattr(aggregation_strategy, "_model_override",
                self._ensemble_detector._model_id)
      ret = (
          ret
          | beam.MapTuple(lambda k, v: ((k, v[0]), v[1]))
          | beam.GroupByKey()
          | f"Run Aggregation Strategy ({model_uuid})" >>
          beam.MapTuple(lambda k, v, agg=aggregation_strategy: (k[0], (
              k[1],
              AnomalyResult(
                  example=v[0].example,
                  prediction=agg.apply([result.prediction for result in v])
              )))).with_output_types(Tuple[Any, Tuple[Any, AnomalyResult]]))

    ret = (
        ret
        | f"Run Threshold Criterion ({model_uuid})" >> _RunThresholdCriterion(
            self._ensemble_detector._model_id,
            self._ensemble_detector._threshold_criterion))

    return ret


class AnomalyDetection(beam.PTransform[beam.PCollection[Tuple[Any, beam.Row]],
                                       beam.PCollection[Tuple[Any,
                                                              AnomalyResult]]]):

  def __init__(
      self,
      detectors: List[AnomalyDetector],
      aggregation_strategy: Optional[AggregationFn] = None,
      threshold_criterion: Optional[ThresholdFn] = None,
      root_model_id: Optional[str] = None,
  ) -> None:

    if root_model_id is None:
      root_model_id = "root"
    self._root = EnsembleAnomalyDetector(
        model_id=root_model_id,
        learners=detectors,
        threshold_criterion=threshold_criterion,
        aggregation_strategy=aggregation_strategy)

    object.__setattr__(self._root, "model_id", root_model_id)

  def maybe_add_key(
      self, element: Tuple[Any, beam.Row]) -> Tuple[Any, Tuple[Any, beam.Row]]:
    key, row = element
    return key, (timestamp.Timestamp.now().micros, row)

  def expand(
      self,
      input: beam.PCollection[Tuple[Any, beam.Row]],
  ) -> beam.PCollection[Tuple[Any, AnomalyResult]]:

    ret: Any = (
        input
        | "Add temp key" >> beam.Map(self.maybe_add_key)
        | _RunEnsembleDetector(self._root))

    remove_temp_key_fn: Callable[[Any, Tuple[Any, AnomalyResult]],
                                 Tuple[Any,
                                       AnomalyResult]] = lambda k, v: (k, v[1])
    ret = ret | "Remove temp key" >> beam.MapTuple(remove_temp_key_fn)

    return ret
