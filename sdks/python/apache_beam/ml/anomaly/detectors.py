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
from typing import Any
from typing import Optional
from typing import List

from apache_beam.ml.anomaly.base import AggregationFunc
from apache_beam.ml.anomaly.base import ThresholdFunc
from apache_beam.ml.anomaly.models import KNOWN_ALGORITHMS


@dataclasses.dataclass(frozen=True)
class AnomalyDetector():
  algorithm: str
  algorithm_args: Optional[dict[str, Any]] = None
  model_id: Optional[str] = None
  features: Optional[List[str]] = None
  target: Optional[str] = None
  threshold_criterion: Optional[ThresholdFunc] = None

  def __post_init__(self):
    canonical_alg = self.algorithm.lower()
    if canonical_alg not in KNOWN_ALGORITHMS:
      raise NotImplementedError(f"Algorithm '{self.algorithm}' not found")

    if not self.model_id:
      object.__setattr__(self, 'model_id', self.algorithm)


@dataclasses.dataclass(frozen=True)
class EnsembleAnomalyDetector(AnomalyDetector):
  n: int = 10
  aggregation_strategy: Optional[AggregationFunc] = None
  learners: Optional[List[AnomalyDetector]] = None

  def __post_init__(self):
    # propagate fields to base class except for id
    field_names = tuple(
        f.name for f in dataclasses.fields(super()) if f.name != "model_id")
    kwargs = {field: getattr(self, field) for field in field_names}

    # set a field (learners) in a frozen dataclass
    if not self.learners:
      super().__setattr__('learners', [])
      for _ in range(self.n):
        self.learners.append(AnomalyDetector(**kwargs))  # type: ignore
    else:
      logging.warning("Setting weak_learners will override all other arguments "
                      "except aggregation_strategy (if set).")
      if self.n != len(self.learners):
        logging.warning("Parameter n will be overwritten with the number of "
                        "weak learners provided to the instantiation.")
        super().__setattr__('n', len(self.learners))

    if not self.model_id:
      object.__setattr__(self, 'model_id', "ensemble")
