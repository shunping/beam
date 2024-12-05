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
    class MyClass():
      pass

    # class is not decorated/registered
    self.assertRaises(AttributeError, lambda: MyClass().to_config())

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

  def test_decorator_key(self):
    # use decorator without parameter
    @configurable
    class MySecondClass():
      pass

    self.assertIn("MySecondClass", KNOWN_CONFIGURABLES)
    self.assertEqual(KNOWN_CONFIGURABLES["MySecondClass"], MySecondClass)
    self.assertTrue(isinstance(MySecondClass(), Configurable))

    # use decorator with key parameter
    @configurable(key="MyThirdKey")
    class MyThirdClass():
      pass

    self.assertIn("MyThirdKey", KNOWN_CONFIGURABLES)
    self.assertEqual(KNOWN_CONFIGURABLES["MyThirdKey"], MyThirdClass)

  def test_init_params_in_configurable(self):
    @configurable
    class MyClassWithInitParams():
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
    class MyCompositeClassWithInitParams():
      def __init__(self, my_class: Optional[MyClassWithInitParams] = None):
        pass

    c = MyCompositeClassWithInitParams(a)
    self.assertEqual(c._init_params, {'my_class': a})

  def test_from_and_to_configurable(self):
    @configurable(on_demand_init=False, just_in_time_init=False)
    @dataclasses.dataclass
    class Product():
      name: str
      price: float

    @configurable(
        key="shopping_entry", on_demand_init=False, just_in_time_init=False)
    class Entry():
      def __init__(self, product: Product, quantity: int = 1):
        self._product = product
        self._quantity = quantity

      def __eq__(self, value: 'Entry') -> bool:
        return self._product == value._product and \
          self._quantity == value._quantity

    @configurable(
        key="shopping_cart", on_demand_init=False, just_in_time_init=False)
    @dataclasses.dataclass
    class ShoppingCart():
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

  def test_on_demand_init(self):
    @configurable(on_demand_init=True, just_in_time_init=False)
    class FooOnDemand():
      counter = 0

      def __init__(self, arg):
        self.my_arg = arg * 10
        type(self).counter += 1

    foo = FooOnDemand(123)
    self.assertEqual(FooOnDemand.counter, 0)
    self.assertIn("_init_params", foo.__dict__)
    self.assertEqual(foo.__dict__["_init_params"], {"arg": 123})

    self.assertNotIn("my_arg", foo.__dict__)
    self.assertRaises(AttributeError, getattr, foo, "my_arg")
    self.assertRaises(AttributeError, lambda: foo.my_arg)
    self.assertRaises(AttributeError, getattr, foo, "unknown_arg")
    self.assertRaises(AttributeError, lambda: foo.unknown_arg)  # type: ignore
    self.assertEqual(FooOnDemand.counter, 0)

    foo_2 = FooOnDemand(456, _run_init=True)  # type: ignore
    self.assertEqual(FooOnDemand.counter, 1)
    self.assertIn("_init_params", foo_2.__dict__)
    self.assertEqual(foo_2.__dict__["_init_params"], {"arg": 456})

    self.assertIn("my_arg", foo_2.__dict__)
    self.assertEqual(foo_2.my_arg, 4560)
    self.assertEqual(FooOnDemand.counter, 1)

  def test_just_in_time_init(self):
    @configurable(on_demand_init=False, just_in_time_init=True)
    class FooJustInTime():
      counter = 0

      def __init__(self, arg):
        self.my_arg = arg * 10
        type(self).counter += 1

    foo = FooJustInTime(321)
    self.assertEqual(FooJustInTime.counter, 0)
    self.assertIn("_init_params", foo.__dict__)
    self.assertEqual(foo.__dict__["_init_params"], {"arg": 321})

    self.assertNotIn("my_arg", foo.__dict__)  # __init__ hasn't been called
    self.assertEqual(FooJustInTime.counter, 0)

    # __init__ is called when trying to accessing an attribute
    self.assertEqual(foo.my_arg, 3210)
    self.assertEqual(FooJustInTime.counter, 1)
    self.assertRaises(AttributeError, lambda: foo.unknown_arg)  # type: ignore
    self.assertEqual(FooJustInTime.counter, 1)

  def test_on_demand_and_just_in_time_init(self):
    @configurable(on_demand_init=True, just_in_time_init=True)
    class FooOnDemandAndJustInTime():
      counter = 0

      def __init__(self, arg):
        self.my_arg = arg * 10
        type(self).counter += 1

    foo = FooOnDemandAndJustInTime(987)
    self.assertEqual(FooOnDemandAndJustInTime.counter, 0)
    self.assertIn("_init_params", foo.__dict__)
    self.assertEqual(foo.__dict__["_init_params"], {"arg": 987})
    self.assertNotIn("my_arg", foo.__dict__)

    self.assertEqual(FooOnDemandAndJustInTime.counter, 0)
    # __init__ is called
    self.assertEqual(foo.my_arg, 9870)
    self.assertEqual(FooOnDemandAndJustInTime.counter, 1)

    # __init__ is called
    foo_2 = FooOnDemandAndJustInTime(789, _run_init=True)  # type: ignore
    self.assertEqual(FooOnDemandAndJustInTime.counter, 2)
    self.assertIn("_init_params", foo_2.__dict__)
    self.assertEqual(foo_2.__dict__["_init_params"], {"arg": 789})

    self.assertEqual(FooOnDemandAndJustInTime.counter, 2)
    # __init__ is NOT called
    self.assertEqual(foo_2.my_arg, 7890)
    self.assertEqual(FooOnDemandAndJustInTime.counter, 2)

  @configurable(on_demand_init=True, just_in_time_init=True)
  class FooForPickle():
    counter = 0

    def __init__(self, arg):
      self.my_arg = arg * 10
      type(self).counter += 1

  def test_on_pickle(self):
    FooForPickle = TestConfigurable.FooForPickle

    import dill
    foo = FooForPickle(456)
    self.assertEqual(FooForPickle.counter, 0)
    new_foo = dill.loads(dill.dumps(foo))
    self.assertEqual(FooForPickle.counter, 0)
    self.assertEqual(new_foo.__dict__, foo.__dict__)

    # Note that pickle does not support classes/functions nested in a function.
    import pickle
    foo = FooForPickle(456)
    self.assertEqual(FooForPickle.counter, 0)
    new_foo = pickle.loads(pickle.dumps(foo))
    self.assertEqual(FooForPickle.counter, 0)
    self.assertEqual(new_foo.__dict__, foo.__dict__)

    import cloudpickle
    foo = FooForPickle(456)
    self.assertEqual(FooForPickle.counter, 0)
    new_foo = cloudpickle.loads(cloudpickle.dumps(foo))
    self.assertEqual(FooForPickle.counter, 0)
    self.assertEqual(new_foo.__dict__, foo.__dict__)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
