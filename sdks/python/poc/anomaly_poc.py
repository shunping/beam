import csv
import logging

import numpy as np

import apache_beam as beam
from apache_beam.ml import anomaly
from apache_beam.options.pipeline_options import PipelineOptions

# INPUT = "/Users/shunping/Projects/beam-dev-python-anomaly/sdks/python/poc/phishing.csv"
INPUT = "/Users/shunping/Projects/beam-dev-python-anomaly/sdks/python/poc/synthetic_data.csv"


def debug_print(element):
  logging.info(str(element))


def synthetic_data():
  # borrowed and modified from river/anomaly/test_lof.py
  n_features = 2
  np.random.seed(12345)
  norm_dist = 0.5 * np.random.rand(500, n_features)
  x_inliers = np.concatenate((norm_dist - 2, norm_dist, norm_dist + 2), axis=0)
  x_outliers = np.concatenate(
      (
          np.random.uniform(low=-4, high=4, size=(20, n_features)),
          np.random.uniform(low=-10, high=-5, size=(10, n_features)),
          np.random.uniform(low=5, high=10, size=(10, n_features)),
      ),
      axis=0,
  )
  x_train = np.concatenate((x_inliers, x_outliers), axis=0)
  ground_truth = np.zeros(len(x_train), dtype=int)
  ground_truth[-len(x_outliers):] = 1
  ground_truth = ground_truth.reshape(len(x_train), 1)
  merged = np.concatenate((x_train, ground_truth), axis=1)
  np.random.shuffle(merged)
  data = [{
      (f"feat_{i + 1}" if i < n_features else "label"): elem[i]
      for i in range(n_features + 1)
  }
          for elem in merged]
  with open("./poc/synthetic_data.csv", "w") as fp:
    writer = csv.DictWriter(fp, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)


def run():
  synthetic_data()

  pipeline_options = PipelineOptions(None)
  with beam.Pipeline(options=pipeline_options) as p:
    data_to_fit = p | beam.io.ReadFromCsv(
        INPUT, splittable=False) | beam.WithKeys(1)
    detectors = []
    detectors.append(
        anomaly.AnomalyDetector(
            algorithm="SAD",
            # id="SAD_1",
            features=["feat_1"],
            target="label",
            #threshold_func=anomaly.FixedThreshold(3),
            threshold_func=anomaly.QuantileThreshold(0.95),
        ))
    detectors.append(
        anomaly.AnomalyDetector(
            algorithm="SAD",
            id="SAD_2",
            algorithm_kwargs={"window_size": 50, "sub_stat": "median"},
            features=["feat_2"],
            target="label"))
    detectors.append(
        anomaly.AnomalyDetector(
            algorithm="MAD",
            algorithm_kwargs={"window_size": 50},
            features=["feat_1"],
            target="label"))
    detectors.append(
        anomaly.AnomalyDetector(
            algorithm="MAD",
            algorithm_kwargs={"window_size": 50},
            features=["feat_2"],
            target="label"))
    # detectors.append(
    #     anomaly.AnomalyDetector(
    #         algorithm="iLOF", features=["feat_1", "feat_2"], target="label"))
    detectors.append(
        anomaly.EnsembleAnomalyDetector(
            n=3,
            algorithm="loda",
            # id="ensemble-loda",
            features=["feat_1", "feat_2"],
            target="label",
            # threshold_func=anomaly.QuantileThreshold(0.95),
        ))
    results = (
        data_to_fit
        | anomaly.AnomalyDetection(
            detectors=detectors,
            with_auc=True,
            aggregation_func=anomaly.LabelAggregation(anomaly.AnyVote()),
        ))

    _ = results | beam.Map(debug_print)


if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  run()
