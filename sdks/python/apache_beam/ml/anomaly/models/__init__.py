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
from typing import Optional

import apache_beam as beam
from apache_beam.ml.anomaly.base import BaseAnomalyModel
from apache_beam.ml.anomaly.models.sad import StandardAbsoluteDeviation
from apache_beam.ml.anomaly.models.mad import MedianAbsoluteDeviation
from apache_beam.ml.anomaly.models.loda import LodaWeakLearner

try:
  from apache_beam.ml.anomaly.models.river import LocalOutlierFactor  # type: ignore
except ImportError:
  logging.warning("Unable to import river model 'LocalOutlierFactor'")
  LocalOutlierFactor = None


class DummyEnsembleAnomalyModel(BaseAnomalyModel):
  def __init__(*args, **kwargs):
    pass

  def learn_one(self, x: beam.Row) -> None:
    raise NotImplementedError("This function should not be called.")

  def score_one(self, x: beam.Row) -> float:
    raise NotImplementedError("This function should not be called.")


KNOWN_ALGORITHMS: dict[str, Optional[type[BaseAnomalyModel]]] = {
    "sad": StandardAbsoluteDeviation,
    "mad": MedianAbsoluteDeviation,
    "loda": LodaWeakLearner,
    "ilof": LocalOutlierFactor,
    "ensemble": DummyEnsembleAnomalyModel,  # a wrapper for ensemble algorithms
}

# Remove unavailable algorithms
KNOWN_ALGORITHMS = {k: v for k, v in KNOWN_ALGORITHMS.items() if v is not None}
