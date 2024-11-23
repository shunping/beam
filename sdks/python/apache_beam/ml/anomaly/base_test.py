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

from apache_beam.ml.anomaly.base import Config
from apache_beam.ml.anomaly.base import AnomalyDetector
from apache_beam.ml.anomaly.base import EnsembleAnomalyDetector


class Dummy(AnomalyDetector):

  def __init__(self, my_arg=None, **kwargs):
    self._my_arg = my_arg
    super().__init__(**kwargs)

  def learn_one(self):
    ...

  def score_one(self):
    ...


# Register the class
AnomalyDetector.register("newly-added-alg", Dummy)


class TestAnomalyDetector(unittest.TestCase):

  def test_unknown_detector(self):
    self.assertRaises(ValueError, AnomalyDetector.from_config,
                      Config(type="unknown"))

  def test_known_detector(self):
    # unregister the class
    AnomalyDetector.unregister("newly-added-alg")

    # Exception occurred when class exists but is not registered yet
    self.assertRaises(ValueError, AnomalyDetector.from_config,
                      Config(type="newly-added-alg"))

    # Register the class
    AnomalyDetector.register("newly-added-alg", Dummy)

    a = AnomalyDetector.from_config(Config(type="newly-added-alg"))
    self.assertTrue(isinstance(a, Dummy))
    self.assertEqual(a._model_id, "newly-added-alg")


class TestEnsembleAnomalyDetector(unittest.TestCase):

  def test_unknown_detector(self):
    self.assertRaises(ValueError, EnsembleAnomalyDetector.from_config,
                      Config(type="unknown"))

  def test_known_detector(self):

    class Dummy(EnsembleAnomalyDetector):

      def learn_one(self):
        ...

      def score_one(self):
        ...

    # Exception occurred when class exists but is not registered yet
    self.assertRaises(ValueError, EnsembleAnomalyDetector.from_config,
                      Config(type="newly-added-ensemble"))

    # Register the class
    EnsembleAnomalyDetector.register("newly-added-ensemble", Dummy)

    a = EnsembleAnomalyDetector.from_config(
        Config(type="newly-added-ensemble", args={"n": 5}))
    self.assertTrue(isinstance(a, Dummy))
    self.assertEqual(a._model_id, "newly-added-ensemble")
    self.assertEqual(a._n, 5)

  def test_known_detector_with_custom_weak_learners(self):
    sub_d1 = AnomalyDetector.from_config(
        Config(
            type="newly-added-alg", args={
                "window_size": 10,
                "model_id": "d1"
            }))
    sub_d2 = AnomalyDetector.from_config(
        Config(
            type="newly-added-alg", args={
                "window_size": 20,
                "model_id": "d2"
            }))

    d = EnsembleAnomalyDetector(n=5, learners=[sub_d1, sub_d2])
    self.assertEqual(d._model_id, "custom")

    assert d._learners
    self.assertEqual(len(d._learners), 2)

    # n is overwritten to the length of learners since learners is provided
    self.assertEqual(d._n, 2)

    self.assertEqual(d._learners[0], sub_d1)
    self.assertEqual(d._learners[1], sub_d2)

    self.assertEqual(d._learners[0]._model_id, "d1")
    self.assertEqual(d._learners[1]._model_id, "d2")

  def test_known_detector_with_custom_weak_learners2(self):
    sub_d1_config = Config(
        type="newly-added-alg", args={
            "my_arg": 10,
            "model_id": "d1"
        })
    sub_d2_config = Config(
        type="newly-added-alg", args={
            "my_arg": 20,
            "model_id": "d2"
        })

    d = EnsembleAnomalyDetector.from_config(
        Config(
            type="custom",
            args={
                "n": 5,
                "learners": [sub_d1_config, sub_d2_config]
            }))

    # d = EnsembleAnomalyDetector(n=5, learners=[sub_d1, sub_d2])
    self.assertEqual(d._model_id, "custom")

    assert d._learners
    self.assertEqual(len(d._learners), 2)

    # n is overwritten to the length of learners since learners is provided
    self.assertEqual(d._n, 2)

    print(d._learners[0].to_config())
    print(sub_d1_config)

    self.assertEqual(d._learners[0].to_config().args["my_arg"],
                     sub_d1_config.args["my_arg"])

    self.assertEqual(d._learners[1].to_config().args["my_arg"],
                     sub_d2_config.args["my_arg"])


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
