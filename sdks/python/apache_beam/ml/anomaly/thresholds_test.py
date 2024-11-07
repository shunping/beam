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
from apache_beam.ml.anomaly.base import AnomalyPrediction
from apache_beam.ml.anomaly.base import AnomalyResult
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to


class TestFixedThreshold(unittest.TestCase):
  def test_threshold(self):
    input = [
        (1, (2, AnomalyResult(beam.Row(x=10), AnomalyPrediction(score=1)))),
        (1, (3, AnomalyResult(beam.Row(x=20), AnomalyPrediction(score=2)))),
        (1, (4, AnomalyResult(beam.Row(x=20), AnomalyPrediction(score=3)))),
    ]
    expected = [
        (
            1,
            (
                2,
                AnomalyResult(
                    beam.Row(x=10),
                    AnomalyPrediction(score=1, label=0, threshold=2)))),
        (
            1,
            (
                3,
                AnomalyResult(
                    beam.Row(x=20),
                    AnomalyPrediction(score=2, label=1, threshold=2)))),
        (
            1,
            (
                4,
                AnomalyResult(
                    beam.Row(x=20),
                    AnomalyPrediction(score=3, label=1, threshold=2)))),
    ]
    with TestPipeline() as p:
      result = (
          p
          | beam.Create(input)
          | beam.ParDo(
              thresholds.StatelessThresholdDoFn(
                  thresholds.FixedThreshold(2, normal_label=0,
                                            outlier_label=1))))

      assert_that(result, equal_to(expected))


class TestQuantileThreshold(unittest.TestCase):
  def test_threshold(self):
    # use the input data with two keys to test stateful threshold function
    input = [
        (1, (2, AnomalyResult(beam.Row(x=10), AnomalyPrediction(score=1)))),
        (1, (3, AnomalyResult(beam.Row(x=20), AnomalyPrediction(score=2)))),
        (1, (4, AnomalyResult(beam.Row(x=30), AnomalyPrediction(score=3)))),
        (2, (2, AnomalyResult(beam.Row(x=40), AnomalyPrediction(score=10)))),
        (2, (3, AnomalyResult(beam.Row(x=50), AnomalyPrediction(score=20)))),
        (2, (4, AnomalyResult(beam.Row(x=60), AnomalyPrediction(score=30)))),
    ]
    expected = [
        (
            1,
            (
                2,
                AnomalyResult(
                    beam.Row(x=10),
                    AnomalyPrediction(score=1, label=1, threshold=1)))),
        (
            1,
            (
                3,
                AnomalyResult(
                    beam.Row(x=20),
                    AnomalyPrediction(score=2, label=1, threshold=1.5)))),
        (
            2,
            (
                2,
                AnomalyResult(
                    beam.Row(x=40),
                    AnomalyPrediction(score=10, label=1, threshold=10)))),
        (
            2,
            (
                3,
                AnomalyResult(
                    beam.Row(x=50),
                    AnomalyPrediction(score=20, label=1, threshold=15)))),
        (
            1,
            (
                4,
                AnomalyResult(
                    beam.Row(x=30),
                    AnomalyPrediction(score=3, label=1, threshold=2)))),
        (
            2,
            (
                4,
                AnomalyResult(
                    beam.Row(x=60),
                    AnomalyPrediction(score=30, label=1, threshold=20)))),
    ]
    with TestPipeline() as p:
      result = (
          p
          | beam.Create(input)
          # use median just for test convenience
          | beam.ParDo(
              thresholds.StatefulThresholdDoFn(
                  thresholds.QuantileThreshold(
                      quantile=0.5, normal_label=0, outlier_label=1))))

      assert_that(result, equal_to(expected))


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
