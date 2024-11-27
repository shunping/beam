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
import inspect
from typing import Any
from typing import List
from typing import Protocol
from typing import TypeVar
from typing import runtime_checkable

KNOWN_CONFIGURABLE = {}


@runtime_checkable
class Configurable(Protocol):
  _key: str
  _init_params: dict[str, Any]


ConfigT = TypeVar('ConfigT', bound=Configurable)


def _register_configurable(cls, key=None, error_if_exists=True) -> None:
  if key is None:
    key = cls.__name__

  if key in KNOWN_CONFIGURABLE and error_if_exists:
    raise ValueError(f"{key} is already registered for configurable")

  KNOWN_CONFIGURABLE[key] = cls

  cls._key = key


def configurable(my_cls=None, /, *, key=None):

  def wrapper(cls):
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
      if not hasattr(self, "_init_params"):
        params = dict(
            zip(
                inspect.signature(original_init).parameters.keys(),
                (None,) + args))
        del params['self']
        params.update(**kwargs)
        self._init_params = params
      original_init(self, *args, **kwargs)

    cls.__init__ = new_init

    _register_configurable(cls, key)

    return cls

  if my_cls is None:
    return wrapper

  return wrapper(my_cls)


@dataclasses.dataclass(frozen=True)
class Config():
  type: str
  args: dict[str, Any] = dataclasses.field(default_factory=dict)

  @staticmethod
  def _from_configurable_helper(v):
    if isinstance(v, Configurable):
      return Config.from_configurable(v)

    if isinstance(v, List):
      return [Config._from_configurable_helper(e) for e in v]

    return v

  @classmethod
  def from_configurable(cls, configurable):
    if getattr(type(configurable), '_key', None) is None:
      raise ValueError(
          f"'{type(configurable).__name__}' not registered as Configurable. "
          f"Call register_configurable({type(configurable).__name__})")

    if not hasattr(configurable, '_init_params'):
      raise ValueError(
          f"{type(configurable).__name__}' not decorated with @configurable.")

    args = {
        k: Config._from_configurable_helper(v)
        for k, v in configurable._init_params.items()
    }

    return Config(type=configurable.__class__._key, args=args)

  @staticmethod
  def _to_configurable_helper(v):
    if isinstance(v, Config):
      return Config.to_configurable(v)

    if isinstance(v, List):
      return [Config._to_configurable_helper(e) for e in v]

    return v

  def to_configurable(self) -> ConfigT:  # type: ignore
    if self.type is None:
      raise ValueError(f"Config type not found in {self}")

    subclass = KNOWN_CONFIGURABLE.get(self.type, None)
    if subclass is None:
      raise ValueError(f"Unknown config type '{self.type}' in {self}")

    args = {k: Config._to_configurable_helper(v) for k, v in self.args.items()}

    return subclass(**args)
