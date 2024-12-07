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
from __future__ import annotations

import dataclasses
import inspect
import logging
from typing import Any
from typing import List
from typing import Protocol
from typing import TypeVar
from typing import Type
from typing import runtime_checkable

KNOWN_CONFIGURABLES = {}

ConfigT = TypeVar('ConfigT', bound='Configurable')


@dataclasses.dataclass(frozen=True)
class Config():
  type: str
  args: dict[str, Any] = dataclasses.field(default_factory=dict)


@runtime_checkable
class Configurable(Protocol):
  _key: str
  _init_params: dict[str, Any]

  @staticmethod
  def _from_config_helper(v):
    if isinstance(v, Config):
      return Configurable.from_config(v)

    if isinstance(v, List):
      return [Configurable._from_config_helper(e) for e in v]

    return v

  @classmethod
  def from_config(cls: Type[ConfigT], config: Config) -> ConfigT:
    if config.type is None:
      raise ValueError(f"Config type not found in {config}")

    subclass: Type[ConfigT] = KNOWN_CONFIGURABLES.get(config.type, None)
    if subclass is None:
      raise ValueError(f"Unknown config type '{config.type}' in {config}")

    args = {
        k: Configurable._from_config_helper(v)
        for k, v in config.args.items()
    }

    return subclass(**args)

  @staticmethod
  def _to_config_helper(v):
    if isinstance(v, Configurable):
      return v.to_config()

    if isinstance(v, List):
      return [Configurable._to_config_helper(e) for e in v]

    return v

  def to_config(self) -> Config:
    if getattr(type(self), '_key', None) is None:
      raise ValueError(
          f"'{type(self).__name__}' not registered as Configurable. "
          f"Decorate ({type(configurable).__name__}) with @configurable")

    args = {k: self._to_config_helper(v) for k, v in self._init_params.items()}

    return Config(type=self.__class__._key, args=args)


def register(cls, key, error_if_exists) -> None:
  if key is None:
    key = cls.__name__

  if key in KNOWN_CONFIGURABLES and error_if_exists:
    raise ValueError(f"{key} is already registered for configurable")

  KNOWN_CONFIGURABLES[key] = cls

  cls._key = key


def configurable(
    my_cls=None,
    /,
    *,
    key=None,
    error_if_exists=True,
    on_demand_init=True,
    just_in_time_init=True):


  def _register_and_track_init_params(cls):
    register(cls, key, error_if_exists)

    original_init = cls.__init__
    class_name = cls.__name__

    def new_init(self, *args, **kwargs):
      self._initialized = False
      self._nested_getattr = False

      if kwargs.get("_run_init", False):
        run_init = True
        del kwargs['_run_init']
      else:
        run_init = False

      if '_init_params' not in self.__dict__:
        params = dict(
            zip(
                inspect.signature(original_init).parameters.keys(),
                (None, ) + args))
        del params['self']
        params.update(**kwargs)
        self._init_params = params

        # If it is not a nested configurable, we choose whether to skip original
        # init call based on options. Otherwise, we always call original init
        # for inner (parent/grandparent/etc) configurable.
        if (on_demand_init and not run_init) or \
            (not on_demand_init and just_in_time_init):
          return

      logging.debug("call original %s.__init__ in new_init", class_name)
      original_init(self, *args, **kwargs)
      self._initialized = True

    # origin_getattribute = cls.__getattribute__

    # def new_get_attribute(self, x):
    #   print(f"call {class_name}.__getattribute__({x})")
    #   return origin_getattribute(self, x)

    #cls.__getattribute__ = new_get_attribute

    def new_getattr(self, name):
      if name == '_nested_getattr' or \
          ('_nested_getattr' in self.__dict__ and self._nested_getattr):
        self._nested_getattr = False
        raise AttributeError(
              f"'{type(self).__name__}' object has no attribute '{name}'")

      # set it before original init, in case getattr is called in original init
      self._nested_getattr = True

      if not self._initialized:
        logging.debug("call original %s.__init__ in new_getattr", class_name)
        original_init(self, **self._init_params)
        self._initialized = True

      try:
        logging.debug("call original %s.getattr in new_getattr", class_name)
        ret = getattr(self, name)
      finally:
        self._nested_getattr = False
      return ret

    if just_in_time_init:
      cls.__getattr__ = new_getattr

    cls.__init__ = new_init
    cls.to_config = Configurable.to_config
    cls._to_config_helper = staticmethod(Configurable._to_config_helper)
    cls.from_config  = classmethod(Configurable.from_config)  # type: ignore
    cls._from_config_helper = staticmethod(Configurable._from_config_helper)
    return cls

  if my_cls is None:
    # support @configurable(...)
    return _register_and_track_init_params

  # support @configurable without arguments
  return _register_and_track_init_params(my_cls)
