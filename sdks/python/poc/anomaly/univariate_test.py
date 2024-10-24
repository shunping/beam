
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
