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
import unittest
from typing import List
from typing import Optional

from apache_beam.ml.anomaly.base import Config
from apache_beam.ml.anomaly.base import Configurable
from apache_beam.ml.anomaly.base import configurable
from apache_beam.ml.anomaly.base import _register_configurable
from apache_beam.ml.anomaly.base import KNOWN_CONFIGURABLE
from apache_beam.ml.anomaly.base import AnomalyDetector
from apache_beam.ml.anomaly.base import EnsembleAnomalyDetector


class TestConfigurable(unittest.TestCase):

  def testRegisterConfigurable(self):

    class MyClass():
      pass

    self.assertNotIn("MyKey", KNOWN_CONFIGURABLE)
    self.assertFalse(isinstance(MyClass, Configurable))
    self.assertFalse(hasattr(MyClass(), '_key'))
    self.assertFalse(hasattr(MyClass(), '_init_params'))

    _register_configurable(MyClass, "MyKey")

    self.assertIn("MyKey", KNOWN_CONFIGURABLE)
    self.assertEqual(KNOWN_CONFIGURABLE["MyKey"], MyClass)
    self.assertFalse(isinstance(MyClass(),
                                Configurable))  # not yet a configurable
    self.assertTrue(hasattr(MyClass(), '_key'))
    self.assertFalse(hasattr(MyClass(), '_init_params'))

    # By default, an error is raised if the key is duplicated
    self.assertRaises(ValueError, _register_configurable, MyClass, "MyKey")

    # But it is ok if a different key is used for the same class
    _register_configurable(MyClass, "MyOtherKey")
    self.assertIn("MyOtherKey", KNOWN_CONFIGURABLE)

    # Or, use a parameter to suppress the error
    _register_configurable(MyClass, "MyClass", error_if_exists=False)

  def testDecorator(self):
    # use decorator without parameter
    @configurable
    class MySecondClass():
      pass

    self.assertIn("MySecondClass", KNOWN_CONFIGURABLE)
    self.assertEqual(KNOWN_CONFIGURABLE["MySecondClass"], MySecondClass)
    self.assertTrue(isinstance(MySecondClass(), Configurable))

    # use decorator with key parameter
    @configurable(key="MyThirdKey")
    class MyThirdClass():
      pass

    self.assertIn("MyThirdKey", KNOWN_CONFIGURABLE)
    self.assertEqual(KNOWN_CONFIGURABLE["MyThirdKey"], MyThirdClass)

  def testInitParamsInConfigurable(self):

    @configurable
    class MyClassWithInitParams():

      def __init__(self, arg_1, arg_2=2, arg_3="3", **kwargs):
        pass

    a = MyClassWithInitParams(10, arg_3="30", arg_4=40)
    self.assertEqual(
        a._init_params,  # type: ignore
        {
            'arg_1': 10,
            'arg_3': '30',
            'arg_4': 40
        })

    # inheritance of configurable
    @configurable
    class MyDerivedClassWithInitParams(MyClassWithInitParams):

      def __init__(self, new_arg_1, new_arg_2=200, new_arg_3="300", **kwargs):
        super().__init__(**kwargs)

    b = MyDerivedClassWithInitParams(
        1000, arg_1=11, arg_2=20, new_arg_2=2000, arg_4=4000)
    self.assertEqual(
        b._init_params,  # type: ignore
        {
            'new_arg_1': 1000,
            'arg_1': 11,
            'arg_2': 20,
            'new_arg_2': 2000,
            'arg_4': 4000
        })

    # composite of configurable
    @configurable
    class MyCompositeClassWithInitParams():
      def __init__(self, my_class: Optional[MyClassWithInitParams] = None):
        pass

    c = MyCompositeClassWithInitParams(a)
    self.assertEqual(
      c._init_params,  # type: ignore
      {
        'my_class': a
      }
    )

  def testToConfigurableAndFromConfigurable(self):

    @configurable
    @dataclasses.dataclass
    class Product():
      name: str
      price: float

    @configurable(key="shopping_entry")
    class Entry():

      def __init__(self, product: Product, quantity: int = 1):
        self._product = product
        self._quantity = quantity

      def __eq__(self, value: 'Entry') -> bool:
        return self._product == value._product and \
          self._quantity == value._quantity

    @configurable(key="shopping_cart")
    @dataclasses.dataclass
    class ShoppingCart():
      user_id: str
      entries: List[Entry]

    orange = Product("orange", 1.0)

    expected_orange_config = Config(
        "Product", args={
            'name': 'orange',
            'price': 1.0
        })
    self.assertEqual(Config.from_configurable(orange), expected_orange_config)
    self.assertEqual(Config.to_configurable(expected_orange_config), orange)

    entry_1 = Entry(product=orange)

    expected_entry_config_1 = Config(
        "shopping_entry", args={
            'product': expected_orange_config,
        })

    self.assertEqual(Config.from_configurable(entry_1), expected_entry_config_1)
    self.assertEqual(Config.to_configurable(expected_entry_config_1), entry_1)

    banana = Product("banana", 0.5)
    expected_banana_config = Config(
        "Product", args={
            'name': 'banana',
            'price': 0.5
        })
    entry_2 = Entry(product=banana, quantity=5)
    expected_entry_config_2 = Config(
        "shopping_entry",
        args={
            'product': expected_banana_config,
            'quantity': 5
        })

    shopping_cart = ShoppingCart(user_id="test", entries=[entry_1, entry_2])
    expected_shopping_cart_config = Config(
        "shopping_cart",
        args={
            "user_id": "test",
            "entries": [expected_entry_config_1, expected_entry_config_2]
        })

    self.assertEqual(
        Config.from_configurable(shopping_cart), expected_shopping_cart_config)
    self.assertEqual(
        Config.to_configurable(expected_shopping_cart_config), shopping_cart)


# class Dummy(AnomalyDetector):

#   def __init__(self, my_arg=None, **kwargs):
#     self._my_arg = my_arg
#     super().__init__(**kwargs)

#   def learn_one(self):
#     ...

#   def score_one(self):
#     ...

# class TestAnomalyDetector(unittest.TestCase):

#   def test_unknown_detector(self):
#     self.assertRaises(ValueError, AnomalyDetector.from_config,
#                       Config(type="unknown"))

#   def test_known_detector(self):
#     AnomalyDetector.unregister("newly-added-alg")

#     # Exception occurred when class exists but is not registered yet
#     self.assertRaises(ValueError, AnomalyDetector.from_config,
#                       Config(type="newly-added-alg"))

#     AnomalyDetector.register("newly-added-alg", Dummy)

#     a = AnomalyDetector.from_config(Config(type="newly-added-alg"))
#     self.assertTrue(isinstance(a, Dummy))
#     self.assertEqual(a._model_id, "newly-added-alg")

#     AnomalyDetector.unregister("newly-added-alg")

# class TestEnsembleAnomalyDetector(unittest.TestCase):

#   def setUp(self) -> None:
#     AnomalyDetector.register("newly-added-alg", Dummy)

#   def tearDown(self) -> None:
#     AnomalyDetector.unregister("newly-added-alg")

#   def test_unknown_detector(self):
#     self.assertRaises(ValueError, EnsembleAnomalyDetector.from_config,
#                       Config(type="unknown"))

#   def test_known_detector(self):

#     class Dummy(EnsembleAnomalyDetector):

#       def learn_one(self):
#         ...

#       def score_one(self):
#         ...

#     # Exception occurred when class exists but is not registered yet
#     self.assertRaises(ValueError, EnsembleAnomalyDetector.from_config,
#                       Config(type="newly-added-ensemble"))

#     EnsembleAnomalyDetector.register("newly-added-ensemble", Dummy)

#     a = EnsembleAnomalyDetector.from_config(
#         Config(type="newly-added-ensemble", args={"n": 5}))
#     self.assertTrue(isinstance(a, Dummy))
#     self.assertEqual(a._model_id, "newly-added-ensemble")
#     self.assertEqual(a._n, 5)

#     EnsembleAnomalyDetector.unregister("newly-added-ensemble")

#   def test_known_detector_with_custom_weak_learners(self):
#     sub_d1 = AnomalyDetector.from_config(
#         Config(
#             type="newly-added-alg", args={
#                 "window_size": 10,
#                 "model_id": "d1"
#             }))
#     sub_d2 = AnomalyDetector.from_config(
#         Config(
#             type="newly-added-alg", args={
#                 "window_size": 20,
#                 "model_id": "d2"
#             }))

#     d = EnsembleAnomalyDetector(n=5, learners=[sub_d1, sub_d2])
#     self.assertEqual(d._model_id, "custom")

#     assert d._learners
#     self.assertEqual(len(d._learners), 2)

#     # n is overwritten to the length of learners since learners is provided
#     self.assertEqual(d._n, 2)

#     self.assertEqual(d._learners[0], sub_d1)
#     self.assertEqual(d._learners[1], sub_d2)

#     self.assertEqual(d._learners[0]._model_id, "d1")
#     self.assertEqual(d._learners[1]._model_id, "d2")

#   def test_known_detector_with_custom_weak_learners2(self):
#     sub_d1_config = Config(
#         type="newly-added-alg", args={
#             "my_arg": 10,
#             "model_id": "d1"
#         })
#     sub_d2_config = Config(
#         type="newly-added-alg", args={
#             "my_arg": 20,
#             "model_id": "d2"
#         })

#     d = EnsembleAnomalyDetector.from_config(
#         Config(
#             type="custom",
#             args={
#                 "n": 5,
#                 "learners": [sub_d1_config, sub_d2_config]
#             }))

#     # d = EnsembleAnomalyDetector(n=5, learners=[sub_d1, sub_d2])
#     self.assertEqual(d._model_id, "custom")

#     assert d._learners
#     self.assertEqual(len(d._learners), 2)

#     # n is overwritten to the length of learners since learners is provided
#     self.assertEqual(d._n, 2)

#     self.assertEqual(d._learners[0].to_config().args["my_arg"],
#                      sub_d1_config.args["my_arg"])

#     self.assertEqual(d._learners[1].to_config().args["my_arg"],
#                      sub_d2_config.args["my_arg"])

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
