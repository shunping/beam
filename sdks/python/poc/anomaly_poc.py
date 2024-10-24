import csv
import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

import numpy as np
from poc import anomaly
from river import optim

# INPUT = "/Users/shunping/Projects/beam-dev-python-anomaly/sdks/python/poc/phishing.csv"
INPUT = "/Users/shunping/Projects/beam-dev-python-anomaly/sdks/python/poc/synthetic_data.csv"

def debug_print(element):
  print(element)


def synthetic_data():
  # borrowed and modified from river/anomaly/test_lof.py
  n_feats = 2
  np.random.seed(12345)
  norm_dist = 0.5 * np.random.rand(500, n_feats)
  x_inliers = np.concatenate((norm_dist - 2, norm_dist, norm_dist + 2), axis=0)
  x_outliers = np.concatenate(
      (
          np.random.uniform(low=-4, high=4, size=(20, n_feats)),
          np.random.uniform(low=-10, high=-5, size=(10, n_feats)),
          np.random.uniform(low=5, high=10, size=(10, n_feats)),
      ),
      axis=0,
  )
  x_train = np.concatenate((x_inliers, x_outliers), axis=0)
  ground_truth = np.zeros(len(x_train), dtype=int)
  ground_truth[-len(x_outliers) :] = 1
  ground_truth = ground_truth.reshape(len(x_train), 1)
  merged = np.concatenate((x_train, ground_truth), axis=1)
  np.random.shuffle(merged)
  data = [{ (f"feature_{i + 1}" if i < n_feats else "label"): elem[i] for i in range(n_feats+1)} for elem in merged]
  with open("./poc/synthetic_data.csv", "w") as fp:
    writer = csv.DictWriter(fp, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)

def run():
  synthetic_data()

  pipeline_options = PipelineOptions(None)
  with beam.Pipeline(options=pipeline_options) as p:
    data_to_fit = p | beam.io.ReadFromCsv(INPUT, splittable=False)
    detectors = []
    detectors.append(anomaly.AnomalyDetector(algorithm="SAD", label="SAD_1", fields=["feature_1"], target="label"))
    #detectors.append(anomaly.AnomalyDetector(algorithm="SAD", label="SAD_2", window_size=50, sub_stat="median", fields=["feature_2"], target="label"))
    #detectors.append(anomaly.AnomalyDetector(algorithm="MAD", window_size=50, fields=["feature_1"], target="label"))
    #detectors.append(anomaly.AnomalyDetector(algorithm="MAD", window_size=50, fields=["feature_2"], target="label"))
    #detectors.append(anomaly.AnomalyDetector(algorithm="iLOF", fields=["feature_1", "feature_2", "feature_3"], target="label"))
    detectors.append(anomaly.EnsembleAnomalyDetector(3, "loda", fields=["feature_1", "feature_2"], target="label"))
    #detectors.append(anomaly.EnsembleAnomalyDetector(3, "loda", fields=["feature_1", "feature_2", "feature_3"], target="label"))
    results = (data_to_fit | anomaly.AnomalyDetection(detectors=detectors))

    _ = results | beam.Map(debug_print)

if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  run()
