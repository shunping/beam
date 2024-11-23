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

import logging
import unittest

from parameterized import parameterized

import apache_beam as beam
from apache_beam.ml.anomaly.aggregations import AverageScore
from apache_beam.ml.anomaly.aggregations import AnyVote
from apache_beam.ml.anomaly.base import AnomalyResult
from apache_beam.ml.anomaly.base import AnomalyPrediction
from apache_beam.ml.anomaly.detectors.sad import StandardAbsoluteDeviation
from apache_beam.ml.anomaly.detectors.loda import Loda
from apache_beam.ml.anomaly.transforms import AnomalyDetection
from apache_beam.ml.anomaly.thresholds import FixedThreshold
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to


class TestAnomalyDetection(unittest.TestCase):

  def setUp(self):
    self._input = [
        (1, beam.Row(x1=1, x2=4)),
        (2, beam.Row(x1=100, x2=5)),  # an row with a different key (key=2)
        (1, beam.Row(x1=2, x2=4)),
        (1, beam.Row(x1=3, x2=5)),
        (1, beam.Row(x1=10, x2=4)),  # outlier in key=1, with respect to x1
        (1, beam.Row(x1=2, x2=10)),  # outlier in key=1, with respect to x2
        (1, beam.Row(x1=3, x2=4)),
    ]

  def test_one_detector(self):
    sad_x1_expected = [
        AnomalyPrediction(model_id='sad_x1', score=0, label=0, threshold=3),
        AnomalyPrediction(model_id='sad_x1', score=0, label=0, threshold=3),
        AnomalyPrediction(model_id='sad_x1', score=0, label=0, threshold=3),
        AnomalyPrediction(
            model_id='sad_x1', score=2.1213203435596424, label=0, threshold=3),
        AnomalyPrediction(model_id='sad_x1', score=8.0, label=1, threshold=3),
        AnomalyPrediction(
            model_id='sad_x1', score=0.4898979485566356, label=0, threshold=3),
        AnomalyPrediction(
            model_id='sad_x1', score=0.16452254913212455, label=0, threshold=3),
    ]
    detectors = []
    detectors.append(
        StandardAbsoluteDeviation(
            features=["x1"],
            threshold_criterion=FixedThreshold(3),
            model_id="sad_x1"))

    with TestPipeline() as p:
      result = (
          p | beam.Create(self._input)
          # TODO: get rid of this conversion between BeamSchema to beam.Row.
          | beam.Map(lambda t: (t[0], beam.Row(**t[1]._asdict())))
          | AnomalyDetection(detectors))
      assert_that(
          result,
          equal_to([(input[0],
                     AnomalyResult(example=input[1], prediction=decision))
                    for input, decision in zip(self._input, sad_x1_expected)]))

  def test_multiple_detectors_without_aggregation(self):
    sad_x1_expected = [
        AnomalyPrediction(model_id='sad_x1', score=0, label=0, threshold=3),
        AnomalyPrediction(model_id='sad_x1', score=0, label=0, threshold=3),
        AnomalyPrediction(model_id='sad_x1', score=0, label=0, threshold=3),
        AnomalyPrediction(
            model_id='sad_x1', score=2.1213203435596424, label=0, threshold=3),
        AnomalyPrediction(model_id='sad_x1', score=8.0, label=1, threshold=3),
        AnomalyPrediction(
            model_id='sad_x1', score=0.4898979485566356, label=0, threshold=3),
        AnomalyPrediction(
            model_id='sad_x1', score=0.16452254913212455, label=0, threshold=3),
    ]
    sad_x2_expected = [
        AnomalyPrediction(model_id='sad_x2', score=0, label=0, threshold=2),
        AnomalyPrediction(model_id='sad_x2', score=0, label=0, threshold=2),
        AnomalyPrediction(model_id='sad_x2', score=0, label=0, threshold=2),
        AnomalyPrediction(model_id='sad_x2', score=0, label=0, threshold=2),
        AnomalyPrediction(
            model_id='sad_x2', score=0.5773502691896252, label=0, threshold=2),
        AnomalyPrediction(model_id='sad_x2', score=11.5, label=1, threshold=2),
        AnomalyPrediction(
            model_id='sad_x2', score=0.5368754921931594, label=0, threshold=2),
    ]

    detectors = []
    detectors.append(
        StandardAbsoluteDeviation(
            features=["x1"],
            threshold_criterion=FixedThreshold(3),
            model_id="sad_x1"))
    detectors.append(
        StandardAbsoluteDeviation(
            features=["x2"],
            threshold_criterion=FixedThreshold(2),
            model_id="sad_x2"))
    with beam.Pipeline() as p:
      result = (
          p | beam.Create(self._input)
          # TODO: get rid of this conversion between BeamSchema to beam.Row.
          | beam.Map(lambda t: (t[0], beam.Row(**t[1]._asdict())))
          | AnomalyDetection(detectors))

      assert_that(
          result,
          equal_to(
              [(input[0], AnomalyResult(example=input[1], prediction=decision))
               for input, decision in zip(self._input, sad_x1_expected)] +
              [(input[0], AnomalyResult(example=input[1], prediction=decision))
               for input, decision in zip(self._input, sad_x2_expected)]))

  def test_multiple_detectors_with_aggregation(self):
    aggregated = [
        AnomalyPrediction(model_id="root", label=0),
        AnomalyPrediction(model_id="root", label=0),
        AnomalyPrediction(model_id="root", label=0),
        AnomalyPrediction(model_id="root", label=0),
        AnomalyPrediction(model_id="root", label=1),
        AnomalyPrediction(model_id="root", label=1),
        AnomalyPrediction(model_id="root", label=0),
    ]

    detectors = []
    detectors.append(
        StandardAbsoluteDeviation(
            features=["x1"],
            threshold_criterion=FixedThreshold(3),
            model_id="sad_x1"))
    detectors.append(
        StandardAbsoluteDeviation(
            features=["x2"],
            threshold_criterion=FixedThreshold(2),
            model_id="sad_x2"))
    with beam.Pipeline() as p:
      result = (
          p | beam.Create(self._input)
          # TODO: get rid of this conversion between BeamSchema to beam.Row.
          | beam.Map(lambda t: (t[0], beam.Row(**t[1]._asdict())))
          | AnomalyDetection(detectors, aggregation_strategy=AnyVote()))

      assert_that(
          result,
          equal_to([(input[0],
                     AnomalyResult(example=input[1], prediction=prediction))
                    for input, prediction in zip(self._input, aggregated)]))

  def test_one_ensemble_detector(self):
    loda = [
        AnomalyPrediction(model_id="loda", score=0),
        AnomalyPrediction(model_id="loda", score=0),
        AnomalyPrediction(model_id="loda", score=0),
        AnomalyPrediction(model_id="loda", score=19.113827924639978),
        AnomalyPrediction(model_id="loda", score=0.63651416837948),
        AnomalyPrediction(model_id="loda", score=10.596634733159407),
        AnomalyPrediction(model_id="loda", score=10.087370092015854),
    ]

    # fix a random seed since loda uses random projections
    import numpy as np
    np.random.seed(12345)

    detectors = []
    detectors.append(Loda(n_init=2, n=3, aggregation_strategy=AverageScore()))
    with beam.Pipeline() as p:
      result = (
          p | beam.Create(self._input)
          # TODO: get rid of this conversion between BeamSchema to beam.Row.
          | beam.Map(lambda t: (t[0], beam.Row(**t[1]._asdict())))
          | AnomalyDetection(detectors))

      assert_that(
          result,
          equal_to([(input[0],
                     AnomalyResult(example=input[1], prediction=decision))
                    for input, decision in zip(self._input, loda)]))


class TestAnomalyDetectionModelId(unittest.TestCase):

  def setUp(self):
    self._input = [(1, beam.Row(x1=1, x2=4))]

  @parameterized.expand([[True, True, None], [True, True, "new_root"],
                         [True, False, None], [True, False, "new_root"],
                         [False, True, None], [False, True, "new_root"],
                         [False, False, None], [False, False, "new_root"]])
  def test_model_id(self, use_threshold, use_aggregation, root_model_id):
    if use_threshold:
      threshold_func = FixedThreshold(3.0)
    else:
      threshold_func = None

    if use_aggregation:
      if use_threshold:
        aggregation_func = AnyVote()
      else:
        aggregation_func = AverageScore()
    else:
      aggregation_func = None

    detectors = []
    detectors.append(
        StandardAbsoluteDeviation(
            features=["x1"], threshold_criterion=threshold_func))
    detectors.append(
        StandardAbsoluteDeviation(
            model_id="sad_x2",
            features=["x2"],
            threshold_criterion=threshold_func))
    detectors.append(
        Loda(
            features=["x1", "x2"],
            aggregation_strategy=AverageScore(),
            threshold_criterion=threshold_func))
    detectors.append(
        Loda(
            model_id="ensemble_2",
            algorithm="loda",
            features=["x1", "x2"],
            aggregation_strategy=AverageScore(),
            threshold_criterion=threshold_func))

    model_id_1 = detectors[0]._model_id
    self.assertEqual(model_id_1, "sad")

    model_id_2 = detectors[1]._model_id
    self.assertEqual(model_id_2, "sad_x2")

    model_id_3 = detectors[2]._model_id
    self.assertEqual(model_id_3, "loda")

    model_id_4 = detectors[3]._model_id
    self.assertEqual(model_id_4, "ensemble_2")

    if use_aggregation:
      # root_model_id is only used in aggregation
      if use_threshold:
        if root_model_id is None:
          predictions = [
              AnomalyPrediction(model_id="root", label=0),
          ]
        else:
          predictions = [
              AnomalyPrediction(model_id=root_model_id, label=0),
          ]
      else:
        if root_model_id is None:
          predictions = [
              AnomalyPrediction(model_id="root", score=0),
          ]
        else:
          predictions = [
              AnomalyPrediction(model_id=root_model_id, score=0),
          ]
    else:
      if use_threshold:
        predictions = [
            AnomalyPrediction(
                model_id=model_id_1, score=0, label=0, threshold=3),
            AnomalyPrediction(
                model_id=model_id_2, score=0, label=0, threshold=3),
            AnomalyPrediction(
                model_id=model_id_3, score=0, label=0, threshold=3),
            AnomalyPrediction(
                model_id=model_id_4, score=0, label=0, threshold=3),
        ]
      else:
        predictions = [
            AnomalyPrediction(model_id=model_id_1, score=0),
            AnomalyPrediction(model_id=model_id_2, score=0),
            AnomalyPrediction(model_id=model_id_3, score=0),
            AnomalyPrediction(model_id=model_id_4, score=0),
        ]

    with beam.Pipeline() as p:
      result = (
          p | beam.Create(self._input)
          # TODO: get rid of this conversion between BeamSchema to beam.Row.
          | beam.Map(lambda t: (t[0], beam.Row(**t[1]._asdict())))
          | AnomalyDetection(
              detectors,
              aggregation_strategy=aggregation_func,
              root_model_id=root_model_id))

      _ = result | beam.Map(print)

      assert_that(
          result,
          equal_to([(input[0],
                     AnomalyResult(example=input[1], prediction=decision))
                    for input, decision in zip(
                        self._input + self._input + self._input +
                        self._input, predictions)]))


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
