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
from typing import Any
from typing import List
from typing import TypeVar

KNOWN_CONFIGURABLES = {}

ConfigT = TypeVar('ConfigT', bound='Configurable')


@dataclasses.dataclass(frozen=True)
class Config():
  type: str
  args: dict[str, Any] = dataclasses.field(default_factory=dict)


class Configurable():
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
  def from_config(cls, config: Config) -> Configurable:
    if config.type is None:
      raise ValueError(f"Config type not found in {config}")

    subclass = KNOWN_CONFIGURABLES.get(config.type, None)
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


def configurable(
    my_cls=None,
    /,
    *,
    key=None,
    error_if_exists=True,
    on_demand_init=True,
    just_in_time_init=True):
  def _register(cls) -> None:
    nonlocal key
    if key is None:
      key = cls.__name__

    if key in KNOWN_CONFIGURABLES and error_if_exists:
      raise ValueError(f"{key} is already registered for configurable")

    KNOWN_CONFIGURABLES[key] = cls

    cls._key = key

  def _register_and_track_init_params(cls):
    _register(cls)

    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
      self._initialized = False
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

      if (on_demand_init and not run_init) or \
          (not on_demand_init and just_in_time_init):
        return

      # set it to True so that if original_init invoke any getattr, it will
      # not enter an infinite loop.
      self._initialized = True
      original_init(self, *args, **kwargs)

    def new_getattr(self, name):
      if '_initialized' in self.__dict__ and not self.__dict__['_initialized']:
        if name == "_init_params":
          raise AttributeError(
              f"'{type(self).__name__}' object has no attribute '{name}'")

        # set it to True so that if original_init invoke any getattr, it will
        # not enter an infinite loop.
        self._initialized = True
        original_init(self, **self._init_params)

      if name not in self.__dict__:
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'")

      return self.__dict__[name]

    if just_in_time_init:
      cls.__getattr__ = new_getattr

    cls.__init__ = new_init
    return cls

  if my_cls is None:
    # support @configurable(...)
    return _register_and_track_init_params

  # support @configurable without arguments
  return _register_and_track_init_params(my_cls)
