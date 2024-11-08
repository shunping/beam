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
import logging
import mock
import unittest

from apache_beam.ml.anomaly import detectors
from apache_beam.ml.anomaly.base import AnomalyModel


class TestAnomalyDetector(unittest.TestCase):
  def setUp(self) -> None:
    self.mocked_model_class = mock.create_autospec(type(AnomalyModel))
    self.my_alg = "newly-added-alg"
    detectors.KNOWN_ALGORITHMS.update({self.my_alg: self.mocked_model_class})

  def test_unknown_detector(self):
    self.assertRaises(
        NotImplementedError, detectors.AnomalyDetector, "unknown-alg")

  def test_known_detector(self):
    d1 = detectors.AnomalyDetector(self.my_alg)
    self.assertEqual(d1.algorithm, self.my_alg)
    assert d1.model_id is not None
    self.assertEqual(d1.model_id, d1.algorithm)

    # model will not be initialized until PTransform expansion
    self.mocked_model_class.assert_not_called()

  def test_known_detector_with_weird_case_alg(self):
    my_alg_with_weird_case = "Newly-ADDED-aLg"
    my_id = "new_id"
    d2 = detectors.AnomalyDetector(my_alg_with_weird_case, model_id=my_id)
    self.assertEqual(d2.algorithm, my_alg_with_weird_case)
    self.assertEqual(d2.model_id, my_id)

    # model will not be initialized until PTransform expansion
    self.mocked_model_class.assert_not_called()


class TestEnsembleAnomalyDetector(unittest.TestCase):
  def setUp(self) -> None:
    self.mocked_model_class = mock.create_autospec(type(AnomalyModel))
    self.my_alg = "newly-added-alg"
    detectors.KNOWN_ALGORITHMS.update({self.my_alg: self.mocked_model_class})

  @staticmethod
  def are_detectors_equal_ignoring_id(
      d1: detectors.AnomalyDetector, d2: detectors.AnomalyDetector):
    field_names = tuple(
        f.name for f in dataclasses.fields(detectors.AnomalyDetector))
    for f in field_names:
      if f == "model_id":
        continue
      if getattr(d1, f) != getattr(d2, f):
        return False
    return True

  def test_unknown_detector(self):
    self.assertRaises(
        NotImplementedError,
        detectors.EnsembleAnomalyDetector,
        "unknown-alg",
    )

  def test_known_detector(self):
    d = detectors.EnsembleAnomalyDetector(self.my_alg)
    self.assertEqual(d.algorithm, self.my_alg)
    self.assertEqual(len(d.learners), 10)  # type: ignore
    for i in range(10):
      self.assertTrue(
          TestEnsembleAnomalyDetector.are_detectors_equal_ignoring_id(
              d.learners[i],  # type: ignore
              detectors.AnomalyDetector(self.my_alg)))
    assert d.model_id is not None
    self.assertEqual(d.model_id, "ensemble")

  def test_known_detector_with_n_and_kwargs(self):
    d = detectors.EnsembleAnomalyDetector(
        self.my_alg, n=5, algorithm_args={"window_size": 50})
    self.assertEqual(d.algorithm, self.my_alg)
    self.assertEqual(len(d.learners), 5)  # type: ignore
    for i in range(5):
      self.assertTrue(
          TestEnsembleAnomalyDetector.are_detectors_equal_ignoring_id(
              d.learners[i],  # type: ignore
              detectors.AnomalyDetector(
                  self.my_alg, algorithm_args={"window_size": 50})))

  def test_known_detector_with_custom_weak_learners(self):
    sub_d1 = detectors.AnomalyDetector(
        self.my_alg, algorithm_args={"window_size": 10})
    sub_d2 = detectors.AnomalyDetector(
        self.my_alg, algorithm_args={"window_size": 20})

    d = detectors.EnsembleAnomalyDetector(
        self.my_alg,
        n=5,
        algorithm_args={"window_size": 50},
        learners=[sub_d1, sub_d2])

    self.assertEqual(d.algorithm, self.my_alg)
    self.assertEqual(len(d.learners), 2)  # type: ignore
    self.assertEqual(d.n, 2)

    self.assertEqual(d.learners[0], sub_d1)  # type: ignore
    self.assertEqual(d.learners[1], sub_d2)  # type: ignore


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
