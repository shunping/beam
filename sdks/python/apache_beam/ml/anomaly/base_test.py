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
from typing import List
from typing import Optional

from apache_beam.ml.anomaly.base import Config
from apache_beam.ml.anomaly.base import Configurable
from apache_beam.ml.anomaly.base import AnomalyDetector
from apache_beam.ml.anomaly.base import EnsembleAnomalyDetector


class TestConfigurable(unittest.TestCase):
  def testCallingRegisterOnBaseClass(self):

    class NewStuff(Configurable):
      pass

    Configurable.register("new-stuff", NewStuff)

    self.assertRaises(ValueError, Configurable.register, "new-stuff", NewStuff)

    Configurable.register("new-stuff", NewStuff, False)

    Configurable.unregister("new-stuff")
    Configurable.register("new-stuff", NewStuff)
    Configurable.unregister("new-stuff")

  def testCallingRegisterOnSubClass(self):

    class NewStuff(Configurable):
      pass

    NewStuff.register("new-stuff2", NewStuff)
    self.assertRaises(ValueError, NewStuff.register, "new-stuff2", NewStuff)

    NewStuff.register("new-stuff2", NewStuff, False)

    NewStuff.unregister("new-stuff2")
    NewStuff.register("new-stuff2", NewStuff)
    NewStuff.unregister("new-stuff2")

  def testCallingRegisterOnDifferentSubClass(self):

    class Func_1(Configurable):
      pass

    class Func_2(Configurable):
      pass

    Func_1.register("my_func", Func_1)
    # Func_1 and Func_2 sharing the same internal mutable class variable.
    # They can not be registered with the same name.
    self.assertRaises(ValueError, Func_2.register, "my_func", Func_2)
    Func_1.unregister("my_func")

  def testToConfigAndFromConfig(self):

    class Person(Configurable):

      def __init__(self, name: str, dad: Optional['Person'],
                   mom: Optional['Person'], friends: Optional[List['Person']]):
        self._name = name
        self._dad = dad
        self._mom = mom
        self._friends = friends

      def __eq__(self, value: 'Person') -> bool:
        return self._name == value._name and self._dad == value._dad and \
          self._mom == value._mom and self._friends == value._friends

    dad = Person("Jack", None, None, None)
    mom = Person("Mary", None, None, None)
    friend_1 = Person("Kay", None, None, None)
    friend_2 = Person("Amy", None, None, None)
    child = Person("Susan", dad, mom, [friend_1, friend_2])

    # Not registered
    self.assertRaises(ValueError, child.to_config)

    Configurable.register("person", Person)
    self.assertEqual(
        dad.to_config(),
        Config(
            "person",
            args={
                'name': 'Jack',
                'friends': None,
                'mom': None,
                'dad': None
            }))
    self.assertEqual(
        mom.to_config(),
        Config(
            "person",
            args={
                'name': 'Mary',
                'friends': None,
                'mom': None,
                'dad': None
            }))
    self.assertEqual(
        friend_1.to_config(),
        Config(
            "person",
            args={
                'name': 'Kay',
                'friends': None,
                'mom': None,
                'dad': None
            }))
    self.assertEqual(
        friend_2.to_config(),
        Config(
            "person",
            args={
                'name': 'Amy',
                'friends': None,
                'mom': None,
                'dad': None
            }))

    self.assertEqual(
        child.to_config(),
        Config(
            "person",
            args={
                'name': 'Susan',
                'friends': [friend_1.to_config(),
                            friend_2.to_config()],
                'mom': mom.to_config(),
                'dad': dad.to_config()
            }))

    dad_dup = Person.from_config(dad.to_config())
    self.assertEqual(dad, dad_dup)
    self.assertEqual(dad.to_config(), dad_dup.to_config())

    child_dup = Person.from_config(child.to_config())
    self.assertEqual(child, child_dup)
    self.assertEqual(child.to_config(), child_dup.to_config())

    Configurable.unregister("person")


class Dummy(AnomalyDetector):

  def __init__(self, my_arg=None, **kwargs):
    self._my_arg = my_arg
    super().__init__(**kwargs)

  def learn_one(self):
    ...

  def score_one(self):
    ...


class TestAnomalyDetector(unittest.TestCase):

  def test_unknown_detector(self):
    self.assertRaises(ValueError, AnomalyDetector.from_config,
                      Config(type="unknown"))

  def test_known_detector(self):
    AnomalyDetector.unregister("newly-added-alg")

    # Exception occurred when class exists but is not registered yet
    self.assertRaises(ValueError, AnomalyDetector.from_config,
                      Config(type="newly-added-alg"))

    AnomalyDetector.register("newly-added-alg", Dummy)

    a = AnomalyDetector.from_config(Config(type="newly-added-alg"))
    self.assertTrue(isinstance(a, Dummy))
    self.assertEqual(a._model_id, "newly-added-alg")

    AnomalyDetector.unregister("newly-added-alg")


class TestEnsembleAnomalyDetector(unittest.TestCase):

  def setUp(self) -> None:
    AnomalyDetector.register("newly-added-alg", Dummy)

  def tearDown(self) -> None:
    AnomalyDetector.unregister("newly-added-alg")

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

    EnsembleAnomalyDetector.register("newly-added-ensemble", Dummy)

    a = EnsembleAnomalyDetector.from_config(
        Config(type="newly-added-ensemble", args={"n": 5}))
    self.assertTrue(isinstance(a, Dummy))
    self.assertEqual(a._model_id, "newly-added-ensemble")
    self.assertEqual(a._n, 5)

    EnsembleAnomalyDetector.unregister("newly-added-ensemble")

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

    self.assertEqual(d._learners[0].to_config().args["my_arg"],
                     sub_d1_config.args["my_arg"])

    self.assertEqual(d._learners[1].to_config().args["my_arg"],
                     sub_d2_config.args["my_arg"])


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
