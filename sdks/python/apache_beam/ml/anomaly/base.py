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

"""Base classes for anomaly detection"""

from abc import ABC, abstractmethod
import dataclasses
from dataclasses import dataclass
from typing import Optional
from typing import Protocol
from typing import Iterable
from typing import Union

import apache_beam as beam

EPSILON = 1e-12

@dataclass(frozen=True)
class AnomalyDecision():
  model: str = ""
  score: float = float('NaN')
  auc: Optional[float] = None
  prediction: Optional[int] = None
  threshold: Optional[float] = None
  info: str = ''


@dataclass(frozen=True)
class AnomalyPrediction():
  data: beam.Row
  decision: AnomalyDecision


class BaseAnomalyModel(ABC):

  @abstractmethod
  def learn_one(self, x: beam.Row) -> None:
    pass

  @abstractmethod
  def score_one(self, x: beam.Row) -> float:
    pass

class BaseThresholdFunc(beam.DoFn):

  @property
  def threshold(self) -> Union[int, float]:
    raise NotImplementedError

  def _update_prediction(self,
                         prediction: AnomalyPrediction) -> AnomalyPrediction:
    pred: int = 0 if prediction.decision.score < self.threshold else 1  # type: ignore
    return dataclasses.replace(
        prediction,
        decision=dataclasses.replace(
            prediction.decision, prediction=pred, threshold=self.threshold))


class AggregationStrategy(Protocol):
  def __call__(self, decisions:Iterable[AnomalyDecision]) -> AnomalyDecision:
    ...
