import logging
import unittest

from pyod.models.loda import LODA

import apache_beam as beam
from apache_beam.ml.anomaly.models.loda import LodaWeakLearner
from apache_beam.ml.anomaly.univariate import SimpleHistogram

import numpy as np


class TestLoda(unittest.TestCase):
  def test_against_pyod(self):
    # input =  np.array([(1, 4), (2, 4), (3, 5), (10, 5), (2, 4), (2, 4.5)])
    # print(input)

    # # loda = LODA(n_random_cuts=10000)
    # # loda.fit(input[:2])
    # # # print(1000 * loda.decision_function(input[2:]))

    # n = 10000
    # my_lodas = []
    # for _ in range(n):
    #   my_lodas.append(LodaWeakLearner(2, SimpleHistogram, {"window_size": 10, "n_bins": 10}))

    # # for j in range(n):
    # #   for i in range(2):
    # #     my_lodas[j].learn_one(beam.Row(x1=input[i][0], x2=input[i][1]))

    # for j in range(n):
    #   for i in range(2):
    #     my_lodas[j].learn_one_np(input[i])

    # for i in range(2, 6):
    #   s = 0
    #   for j in range(n):
    #     s += my_lodas[j].score_one_np(input[i])
    #   print(s/n)
    pass


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  unittest.main()
