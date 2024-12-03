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

KNOWN_CONFIGURABLE = {}

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

    subclass = KNOWN_CONFIGURABLE.get(config.type, None)
    if subclass is None:
      raise ValueError(f"Unknown config type '{config.type}' in {config}")

    args = {k: Configurable._from_config_helper(v) for k, v in config.args.items()}

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

    args = {
        k: self._to_config_helper(v)
        for k,
        v in self._init_params.items()
    }

    return Config(type=self.__class__._key, args=args)


def configurable(my_cls=None, /, *, key=None, error_if_exists=True, lazy_init=True):
  def _register(cls) -> None:
    nonlocal key
    if key is None:
      key = cls.__name__

    if key in KNOWN_CONFIGURABLE and error_if_exists:
      raise ValueError(f"{key} is already registered for configurable")

    KNOWN_CONFIGURABLE[key] = cls

    cls._key = key

  def _register_and_track_init_params(cls):
    _register(cls)

    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
      if not hasattr(self, "_init_params"):
        params = dict(
            zip(
                inspect.signature(original_init).parameters.keys(),
                (None, ) + args))
        del params['self']
        params.update(**kwargs)
        self._init_params = params

      if "_run_init" in kwargs:
        run_init = True
        del kwargs['_run_init']
      else:
        run_init = False

      if lazy_init and not run_init:
        if self._init_params is not None:
          for k, v in self._init_params.items():
            setattr(self, f"_{k}", v)
        return

      original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls

  if my_cls is None:
    return _register_and_track_init_params

  return  _register_and_track_init_params(my_cls)
