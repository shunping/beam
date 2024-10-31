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

import river.anomaly

import apache_beam as beam
from apache_beam.ml.anomaly.base import BaseAnomalyModel


class RiverAnomalyModel(BaseAnomalyModel):
  def __init__(self):
    self._river_model = None

  def learn_one(self, x: beam.Row):
    self._river_model.learn_one(x.__dict__)  # type: ignore

  def score_one(self, x: beam.Row):
    return self._river_model.score_one(x.__dict__)  # type: ignore


class LocalOutlierFactor(RiverAnomalyModel):
  def __init__(self, *args, **kwargs):
    self._river_model = river.anomaly.LocalOutlierFactor(*args, **kwargs)
