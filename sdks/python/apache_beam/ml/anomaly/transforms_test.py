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

import apache_beam as beam
from apache_beam.ml.anomaly.aggregations import AnyVote
from apache_beam.ml.anomaly.base import AnomalyResult
from apache_beam.ml.anomaly.base import AnomalyPrediction
from apache_beam.ml.anomaly.detectors import AnomalyDetector
from apache_beam.ml.anomaly.detectors import EnsembleAnomalyDetector
from apache_beam.ml.anomaly.transforms import AnomalyDetection
from apache_beam.ml.anomaly.thresholds import FixedThreshold
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
            model_id='sad_x1', score=2.1213203435596424, label=0,
            threshold=3),
        AnomalyPrediction(model_id='sad_x1', score=8.0, label=1, threshold=3),
        AnomalyPrediction(
            model_id='sad_x1', score=0.4898979485566356, label=0,
            threshold=3),
        AnomalyPrediction(
            model_id='sad_x1', score=0.16452254913212455, label=0,
            threshold=3),
    ]
    detectors = []
    detectors.append(
        AnomalyDetector(
            algorithm="SAD",
            features=["x1"],
            threshold_criterion=FixedThreshold(3),
            id="sad_x1"))

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
                    for input, decision in zip(self._input, sad_x1_expected)]))

  def test_multiple_detectors_without_aggregation(self):
    sad_x1_expected = [
        AnomalyPrediction(model_id='sad_x1', score=0, label=0, threshold=3),
        AnomalyPrediction(model_id='sad_x1', score=0, label=0, threshold=3),
        AnomalyPrediction(model_id='sad_x1', score=0, label=0, threshold=3),
        AnomalyPrediction(
            model_id='sad_x1', score=2.1213203435596424, label=0,
            threshold=3),
        AnomalyPrediction(model_id='sad_x1', score=8.0, label=1, threshold=3),
        AnomalyPrediction(
            model_id='sad_x1', score=0.4898979485566356, label=0,
            threshold=3),
        AnomalyPrediction(
            model_id='sad_x1', score=0.16452254913212455, label=0,
            threshold=3),
    ]
    sad_x2_expected = [
        AnomalyPrediction(model_id='sad_x2', score=0, label=0, threshold=2),
        AnomalyPrediction(model_id='sad_x2', score=0, label=0, threshold=2),
        AnomalyPrediction(model_id='sad_x2', score=0, label=0, threshold=2),
        AnomalyPrediction(model_id='sad_x2', score=0, label=0, threshold=2),
        AnomalyPrediction(
            model_id='sad_x2', score=0.5773502691896252, label=0,
            threshold=2),
        AnomalyPrediction(model_id='sad_x2', score=11.5, label=1, threshold=2),
        AnomalyPrediction(
            model_id='sad_x2',
            score=0.5368754921931594,
            label=0,
            threshold=2),
    ]

    detectors = []
    detectors.append(
        AnomalyDetector(
            algorithm="SAD",
            features=["x1"],
            threshold_criterion=FixedThreshold(3),
            id="sad_x1"))
    detectors.append(
        AnomalyDetector(
            algorithm="SAD",
            features=["x2"],
            threshold_criterion=FixedThreshold(2),
            id="sad_x2"))
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
        AnomalyPrediction(label=0),
        AnomalyPrediction(label=0),
        AnomalyPrediction(label=0),
        AnomalyPrediction(label=0),
        AnomalyPrediction(label=1),
        AnomalyPrediction(label=1),
        AnomalyPrediction(label=0),
    ]

    detectors = []
    detectors.append(
        AnomalyDetector(
            algorithm="SAD",
            features=["x1"],
            threshold_criterion=FixedThreshold(3),
            id="sad_x1"))
    detectors.append(
        AnomalyDetector(
            algorithm="SAD",
            features=["x2"],
            threshold_criterion=FixedThreshold(2),
            id="sad_x2"))
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
        AnomalyPrediction(score=0),
        AnomalyPrediction(score=0),
        AnomalyPrediction(score=0),
        AnomalyPrediction(score=19.113827924639978),
        AnomalyPrediction(score=0.63651416837948),
        AnomalyPrediction(score=10.596634733159407),
        AnomalyPrediction(score=10.087370092015854),
    ]

    # fix a random seed since loda uses random projections
    import numpy as np
    np.random.seed(12345)

    detectors = []
    detectors.append(
        EnsembleAnomalyDetector(
            algorithm="loda", algorithm_kwargs={"n_init": 2}, n=3))
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


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
