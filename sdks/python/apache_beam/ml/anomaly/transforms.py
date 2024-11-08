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
from typing import Generic
from typing import Iterable
from typing import Tuple
from typing import Optional
from typing import TypeVar

import apache_beam as beam
from apache_beam.coders import DillCoder
from apache_beam.ml.anomaly import thresholds
from apache_beam.ml.anomaly.base import AnomalyModel
from apache_beam.ml.anomaly.base import AnomalyPrediction
from apache_beam.ml.anomaly.base import AnomalyResult
from apache_beam.ml.anomaly.base import AggregationFunc
from apache_beam.ml.anomaly.base import ExampleT
from apache_beam.ml.anomaly.base import LabelT
from apache_beam.ml.anomaly.base import ScoreT
from apache_beam.ml.anomaly.detectors import AnomalyDetector
from apache_beam.ml.anomaly.detectors import EnsembleAnomalyDetector
from apache_beam.ml.anomaly.models import KNOWN_ALGORITHMS
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec
from apache_beam.utils import timestamp

KeyT = TypeVar('KeyT')


class _ScoreAndLearn(beam.DoFn, Generic[ExampleT, ScoreT, LabelT]):
  MODEL_STATE_INDEX = ReadModifyWriteStateSpec('saved_model', DillCoder())

  def __init__(self, detector: AnomalyDetector):
    self._detector = detector
    self._canonical_alg = self._detector.algorithm.lower()
    if not self._canonical_alg in KNOWN_ALGORITHMS:
      raise NotImplementedError(f"algorithm '{detector.algorithm}' not found")

  def score_and_learn(self, data):
    x = self._underlying.get_x(data)
    y_pred = self._underlying.score_one(x)
    self._underlying.learn_one(x)
    return y_pred

  def process(
      self,
      element: Tuple[KeyT, Tuple[Any, ExampleT]],
      model_state=beam.DoFn.StateParam(MODEL_STATE_INDEX),
      **kwargs) -> Iterable[Tuple[KeyT, Tuple[Any, AnomalyResult]]]:

    k1, (k2, data) = element
    self._underlying = model_state.read()  # type: ignore
    if self._underlying is None:
      model_class = KNOWN_ALGORITHMS[self._canonical_alg]
      assert model_class is not None, "model {self._canonical_alg} not found"
      kwargs = self._detector.algorithm_args if self._detector.algorithm_args is not None else {}
      kwargs.update({
          "features": self._detector.features, "target": self._detector.target
      })
      self._underlying: AnomalyModel[ExampleT, ScoreT] = model_class(**kwargs)

    yield k1, (k2,
               AnomalyResult(
                   example=data,
                   prediction=AnomalyPrediction[ScoreT, LabelT](
                       model_id=self._detector.model_id,
                       score=self.score_and_learn(data))))

    model_state.write(self._underlying)  # type: ignore


class _RunDetectors(
    beam.PTransform[beam.PCollection[Tuple[KeyT, Tuple[Any, ExampleT]]],
                    beam.PCollection[Tuple[KeyT,
                                           Tuple[Any,
                                                 AnomalyResult[ExampleT,
                                                               ScoreT,
                                                               LabelT]]]]],
    Generic[KeyT, ExampleT, ScoreT, LabelT]):
  def __init__(
      self,
      model_id: Optional[str],
      detectors: Iterable[AnomalyDetector[ScoreT, LabelT]],
      aggregation_strategy: Optional[AggregationFunc] = None):
    self._model_id = model_id
    self._detectors = detectors
    self._aggregation_strategy = aggregation_strategy

  def expand(
      self, input: beam.PCollection[Tuple[KeyT, Tuple[Any, ExampleT]]]
  ) -> beam.PCollection[Tuple[KeyT, Tuple[Any, AnomalyResult]]]:
    model_results = []
    for detector in self._detectors:
      if isinstance(detector, EnsembleAnomalyDetector):
        if detector.learners:
          score_result = (
              input | f"Run detectors under {detector}" >> _RunDetectors(
                  detector.model_id,
                  detector.learners,
                  detector.aggregation_strategy))
        else:
          raise ValueError("No learners found at {self._model_id}")
      else:
        score_result = (
            input
            | f"Reshuffle ({detector})" >> beam.Reshuffle()
            | f"Score and learn ({detector})" >> beam.ParDo(
                _ScoreAndLearn(detector)).with_output_types(
                    Tuple[KeyT, Tuple[Any, AnomalyResult]]))

      if detector.threshold_criterion:
        if detector.threshold_criterion.is_stateful:
          model_results.append(
              score_result
              | f"Run stateful threshold function ({detector})" >> beam.ParDo(
                  thresholds.StatefulThresholdDoFn(
                      detector.threshold_criterion)))
        else:
          model_results.append(
              score_result
              | f"Run stateless threshold function ({detector})" >> beam.ParDo(
                  thresholds.StatelessThresholdDoFn(
                      detector.threshold_criterion)))
      else:
        model_results.append(score_result)

    merged = model_results | f"Flatten {self._model_id}" >> beam.Flatten()

    ret = merged
    if self._aggregation_strategy is not None:
      # if no model_override is set in the aggregation function, use
      # model id locally in the instance
      if getattr(self._aggregation_strategy, "_model_override") is None:
        setattr(self._aggregation_strategy, "_model_override", self._model_id)
      ret = (
          ret
          | beam.MapTuple(lambda k, v: ((k, v[0]), v[1]))
          | beam.GroupByKey()
          | beam.MapTuple(
              lambda k,
              v,
              agg=self._aggregation_strategy: (
                  k[0],
                  (
                      k[1],
                      AnomalyResult(
                          example=v[0].example,
                          prediction=agg([result.prediction for result in v]))))
          ).with_output_types(Tuple[KeyT, Tuple[Any, AnomalyResult]]))

    return ret


class AnomalyDetection(beam.PTransform[beam.PCollection[Tuple[KeyT, ExampleT]],
                                       beam.PCollection[Tuple[KeyT,
                                                              AnomalyResult]]],
                       Generic[KeyT, ExampleT, ScoreT, LabelT]):
  def __init__(
      self,
      detectors: Iterable[AnomalyDetector[ScoreT, LabelT]],
      aggregation_strategy: Optional[AggregationFunc[ScoreT, LabelT]] = None,
      root_model_id: Optional[str] = None,
  ) -> None:
    self._detectors = detectors
    self._aggregation_strategy = aggregation_strategy
    self._root_model_id = root_model_id

  def maybe_add_key(
      self, element: Tuple[KeyT,
                           ExampleT]) -> Tuple[KeyT, Tuple[Any, ExampleT]]:
    key, row = element
    return key, (timestamp.Timestamp.now().micros, row)

  def expand(
      self,
      input: beam.PCollection[Tuple[KeyT, ExampleT]],
  ) -> beam.PCollection[Tuple[KeyT, AnomalyResult]]:

    assert self._detectors is not None

    ret = (
        input
        | "Add temp key" >> beam.Map(self.maybe_add_key)
        | _RunDetectors(self._root_model_id, self._detectors, self._aggregation_strategy))

    remove_temp_key_func: Callable[
        [KeyT, Tuple[Any, AnomalyResult[ExampleT, ScoreT, LabelT]]],
        Tuple[KeyT, AnomalyResult]] = lambda k, v: (k, v[1])
    ret |= beam.MapTuple(remove_temp_key_func)

    return ret  # type: ignore
