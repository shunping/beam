import csv
import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

import numpy as np
from poc import anomaly
from river import optim

# INPUT = "/Users/shunping/Projects/beam-dev-python-anomaly/sdks/python/poc/phishing.csv"
INPUT = "/Users/shunping/Projects/beam-dev-python-anomaly/sdks/python/poc/synthetic_data.csv"

def extract_feature_and_target(element):
  data = element._asdict()
  #target = 'is_phishing'
  target = 'label'
  x = {k: v for k, v in data.items() if k != target}
  y = bool(data[target])
  return x, y

def debug_print(element):
  print(element)


def synthetic_data():
  # borrowed and modified from river/anomaly/test_lof.py
  np.random.seed(12345)
  norm_dist = 0.5 * np.random.rand(100, 2)
  x_inliers = np.concatenate((norm_dist - 2, norm_dist, norm_dist + 2), axis=0)
  x_outliers = np.concatenate(
      (
          np.random.uniform(low=-4, high=4, size=(20, 2)),
          np.random.uniform(low=-10, high=-5, size=(10, 2)),
          np.random.uniform(low=5, high=10, size=(10, 2)),
      ),
      axis=0,
  )
  x_train = np.concatenate((x_inliers, x_outliers), axis=0)
  ground_truth = np.zeros(len(x_train), dtype=int)
  ground_truth[-len(x_outliers) :] = 1
  ground_truth = ground_truth.reshape(len(x_train), 1)
  merged = np.concatenate((x_train, ground_truth), axis=1)
  np.random.shuffle(merged)
  data = [{ (f"feature_{i + 1}" if i < 2 else "label"): elem[i] for i in range(3)} for elem in merged]
  with open("./poc/synthetic_data.csv", "w") as fp:
    writer = csv.DictWriter(fp, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)

def run():
  # synthetic_data()

  pipeline_options = PipelineOptions(None)
  with beam.Pipeline(options=pipeline_options) as p:
    data_to_fit = (
        p
        | beam.io.ReadFromCsv(INPUT, splittable=False)
        | beam.Map(extract_feature_and_target))

    # detector1 = anomaly.AnomalyDetector(algorithm="NaiveBayesian")
    # detector2 = anomaly.AnomalyDetector(algorithm="LogisticRegression")
    detector1 = anomaly.AnomalyDetector(algorithm="IncrementalLocalOutlierFactor")
    detector2 = anomaly.AnomalyDetector(algorithm="HalfSpaceTree")
    # detector3 = anomaly.AnomalyDetector(algorithm="OneClassSVM", optimizer=optim.SGD(1e-2), nu=0.5)

    results = (data_to_fit
               | anomaly.AnomalyDetection(detectors=[detector1,
                                                     detector2]))

    _ = results | beam.Map(debug_print)

if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  run()
