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
import math
import unittest

from poc.anomaly import univariate

class MeanTest(unittest.TestCase):
  def test_mean(self):
    mt = univariate.SimpleMeanTracker(3)
    self.assertTrue(math.isnan(mt.get()))
    mt.push(1)
    self.assertEqual(mt.get(), 1)
    mt.push(2)
    self.assertEqual(mt.get(), 1.5)
    mt.push(3)
    self.assertEqual(mt.get(), 2.0)
    mt.push(10)
    self.assertEqual(mt.get(), 5.0)

  def test_rolling_mean(self):
    mt = univariate.RollingMeanTracker(3)
    self.assertTrue(math.isnan(mt.get()))
    mt.push(1)
    self.assertEqual(mt.get(), 1)
    mt.push(2)
    self.assertEqual(mt.get(), 1.5)
    mt.push(3)
    self.assertEqual(mt.get(), 2.0)
    mt.push(10)
    self.assertEqual(mt.get(), 5.0)


class MedianTest(unittest.TestCase):
  def test_median(self):
    mt = univariate.SimpleMedianTracker(3)
    self.assertTrue(math.isnan(mt.get()))
    mt.push(1)
    self.assertEqual(mt.get(), 1)
    mt.push(2)
    self.assertEqual(mt.get(), 1.5)
    mt.push(3)
    self.assertEqual(mt.get(), 2.0)
    mt.push(10)
    self.assertEqual(mt.get(), 3.0)


class StdevTest(unittest.TestCase):
  def test_stdev(self):
    mt = univariate.SimpleStdevTracker(3)
    self.assertTrue(math.isnan(mt.get()))
    mt.push(1)
    self.assertTrue(math.isnan(mt.get()))
    mt.push(2)
    self.assertEqual(mt.get(), 0.7071067811865476)
    mt.push(3)
    self.assertEqual(mt.get(), 1.0)
    mt.push(10)
    self.assertEqual(mt.get(), 4.358898943540674)

  def test_rolling_stdev(self):
    mt = univariate.RollingStdevTracker(3)
    self.assertTrue(math.isnan(mt.get()))
    mt.push(1)
    self.assertTrue(math.isnan(mt.get()))
    mt.push(2)
    self.assertEqual(mt.get(), 0.7071067811865476)
    mt.push(3)
    self.assertEqual(mt.get(), 1.0)
    mt.push(10)
    self.assertEqual(mt.get(), 4.358898943540674)


class MADTest(unittest.TestCase):
  def test_mad(self):
    mt = univariate.SimpleMADTracker(3)
    mt.push(1)
    mt.push(2)
    mt.push(3)
    self.assertEqual(mt.get(), 1)
    mt.push(10)
    self.assertEqual(mt.get(), 1)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
