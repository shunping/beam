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
from typing import List
from typing import Optional
import unittest

from apache_beam.ml.anomaly.configurable import Config
from apache_beam.ml.anomaly.configurable import Configurable
from apache_beam.ml.anomaly.configurable import configurable
from apache_beam.ml.anomaly.configurable import KNOWN_CONFIGURABLES


class TestConfigurable(unittest.TestCase):
  def test_register_configurable(self):
    class MyClass(Configurable):
      pass

    # class is not decorated/registered
    self.assertRaises(ValueError, MyClass().to_config)

    self.assertNotIn("MyKey", KNOWN_CONFIGURABLES)

    MyClass = configurable(key="MyKey")(MyClass)

    self.assertIn("MyKey", KNOWN_CONFIGURABLES)
    self.assertEqual(KNOWN_CONFIGURABLES["MyKey"], MyClass)

    # By default, an error is raised if the key is duplicated
    self.assertRaises(ValueError, configurable(key="MyKey"), MyClass)

    # But it is ok if a different key is used for the same class
    _ = configurable(key="MyOtherKey")(MyClass)
    self.assertIn("MyOtherKey", KNOWN_CONFIGURABLES)

    # Or, use a parameter to suppress the error
    configurable(key="MyKey", error_if_exists=False)(MyClass)

  def test_decorator(self):
    # use decorator without parameter
    @configurable
    class MySecondClass(Configurable):
      pass

    self.assertIn("MySecondClass", KNOWN_CONFIGURABLES)
    self.assertEqual(KNOWN_CONFIGURABLES["MySecondClass"], MySecondClass)
    self.assertTrue(isinstance(MySecondClass(), Configurable))

    # use decorator with key parameter
    @configurable(key="MyThirdKey")
    class MyThirdClass(Configurable):
      pass

    self.assertIn("MyThirdKey", KNOWN_CONFIGURABLES)
    self.assertEqual(KNOWN_CONFIGURABLES["MyThirdKey"], MyThirdClass)

  def test_init_params_in_configurable(self):
    @configurable
    class MyClassWithInitParams(Configurable):
      def __init__(self, arg_1, arg_2=2, arg_3="3", **kwargs):
        pass

    a = MyClassWithInitParams(10, arg_3="30", arg_4=40)
    self.assertEqual(a._init_params, {'arg_1': 10, 'arg_3': '30', 'arg_4': 40})

    # inheritance of configurable
    @configurable
    class MyDerivedClassWithInitParams(MyClassWithInitParams):
      def __init__(self, new_arg_1, new_arg_2=200, new_arg_3="300", **kwargs):
        super().__init__(**kwargs)

    b = MyDerivedClassWithInitParams(
        1000, arg_1=11, arg_2=20, new_arg_2=2000, arg_4=4000)
    self.assertEqual(
        b._init_params,
        {
            'new_arg_1': 1000,
            'arg_1': 11,
            'arg_2': 20,
            'new_arg_2': 2000,
            'arg_4': 4000
        })

    # composite of configurable
    @configurable
    class MyCompositeClassWithInitParams(Configurable):
      def __init__(self, my_class: Optional[MyClassWithInitParams] = None):
        pass

    c = MyCompositeClassWithInitParams(a)
    self.assertEqual(c._init_params, {'my_class': a})

  def test_from_and_to_configurable(self):
    @configurable(lazy_init=False)
    @dataclasses.dataclass
    class Product(Configurable):
      name: str
      price: float

    @configurable(key="shopping_entry", lazy_init=False)
    class Entry(Configurable):
      def __init__(self, product: Product, quantity: int = 1):
        self._product = product
        self._quantity = quantity

      def __eq__(self, value: 'Entry') -> bool:
        return self._product == value._product and \
          self._quantity == value._quantity

    @configurable(key="shopping_cart", lazy_init=False)
    @dataclasses.dataclass
    class ShoppingCart(Configurable):
      user_id: str
      entries: List[Entry]

    orange = Product("orange", 1.0)

    expected_orange_config = Config(
        "Product", args={
            'name': 'orange', 'price': 1.0
        })
    self.assertEqual(orange.to_config(), expected_orange_config)
    self.assertEqual(Configurable.from_config(expected_orange_config), orange)

    entry_1 = Entry(product=orange)

    expected_entry_config_1 = Config(
        "shopping_entry", args={
            'product': expected_orange_config,
        })

    self.assertEqual(entry_1.to_config(), expected_entry_config_1)
    self.assertEqual(Configurable.from_config(expected_entry_config_1), entry_1)

    banana = Product("banana", 0.5)
    expected_banana_config = Config(
        "Product", args={
            'name': 'banana', 'price': 0.5
        })
    entry_2 = Entry(product=banana, quantity=5)
    expected_entry_config_2 = Config(
        "shopping_entry",
        args={
            'product': expected_banana_config, 'quantity': 5
        })

    shopping_cart = ShoppingCart(user_id="test", entries=[entry_1, entry_2])
    expected_shopping_cart_config = Config(
        "shopping_cart",
        args={
            "user_id": "test",
            "entries": [expected_entry_config_1, expected_entry_config_2]
        })

    self.assertEqual(shopping_cart.to_config(), expected_shopping_cart_config)
    self.assertEqual(
        Configurable.from_config(expected_shopping_cart_config), shopping_cart)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
