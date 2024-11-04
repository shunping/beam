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
from apache_beam.ml.anomaly import thresholds
from apache_beam.ml.anomaly.base import AnomalyDecision
from apache_beam.ml.anomaly.base import AnomalyPrediction
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to


class TestFixedThreshold(unittest.TestCase):

  def test_threshold(self):
    input = [
        (1, (2, AnomalyPrediction(beam.Row(x=10), AnomalyDecision(score=1)))),
        (1, (3, AnomalyPrediction(beam.Row(x=20), AnomalyDecision(score=2)))),
        (1, (4, AnomalyPrediction(beam.Row(x=20), AnomalyDecision(score=3)))),
    ]
    expected = [
        (1, (2,
             AnomalyPrediction(
                 beam.Row(x=10),
                 AnomalyDecision(score=1, prediction=0, threshold=2)))),
        (1, (3,
             AnomalyPrediction(
                 beam.Row(x=20),
                 AnomalyDecision(score=2, prediction=1, threshold=2)))),
        (1, (4,
             AnomalyPrediction(
                 beam.Row(x=20),
                 AnomalyDecision(score=3, prediction=1, threshold=2)))),
    ]
    with TestPipeline() as p:
      result = (
          p
          | beam.Create(input)
          | beam.ParDo(thresholds.FixedThreshold(2)))

      assert_that(result, equal_to(expected))


class TestQuantileThreshold(unittest.TestCase):

  def test_threshold(self):
    # use the input data with two keys to test stateful threshold function
    input = [
        (1, (2, AnomalyPrediction(beam.Row(x=10), AnomalyDecision(score=1)))),
        (1, (3, AnomalyPrediction(beam.Row(x=20), AnomalyDecision(score=2)))),
        (1, (4, AnomalyPrediction(beam.Row(x=30), AnomalyDecision(score=3)))),
        (2, (2, AnomalyPrediction(beam.Row(x=40), AnomalyDecision(score=10)))),
        (2, (3, AnomalyPrediction(beam.Row(x=50), AnomalyDecision(score=20)))),
        (2, (4, AnomalyPrediction(beam.Row(x=60), AnomalyDecision(score=30)))),
    ]
    expected = [
        (1, (2,
             AnomalyPrediction(
                 beam.Row(x=10),
                 AnomalyDecision(score=1, prediction=1, threshold=1)))),
        (1, (3,
             AnomalyPrediction(
                 beam.Row(x=20),
                 AnomalyDecision(score=2, prediction=1, threshold=1.5)))),
        (2, (2,
             AnomalyPrediction(
                 beam.Row(x=40),
                 AnomalyDecision(score=10, prediction=1, threshold=10)))),
        (2, (3,
             AnomalyPrediction(
                 beam.Row(x=50),
                 AnomalyDecision(score=20, prediction=1, threshold=15)))),
        (1, (4,
             AnomalyPrediction(
                 beam.Row(x=30),
                 AnomalyDecision(score=3, prediction=1, threshold=2)))),
        (2, (4,
             AnomalyPrediction(
                 beam.Row(x=60),
                 AnomalyDecision(score=30, prediction=1, threshold=20)))),
    ]
    with TestPipeline() as p:
      result = (
          p
          | beam.Create(input)
          # use median just for test convenience
          | beam.ParDo(thresholds.QuantileThreshold(0.5)))

      assert_that(result, equal_to(expected))


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
