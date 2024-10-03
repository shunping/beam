import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from poc import anomaly

INPUT = "/Users/shunping/Projects/beam-dev-python-anomaly/sdks/python/poc/phishing.csv"

def extract_feature_and_target(element):
  data = element._asdict()
  target = 'is_phishing'
  x = {k: v for k, v in data.items() if k != target}
  y = bool(data[target])
  return x, y

def debug_print(element):
  print(element)

def run():
  pipeline_options = PipelineOptions(None)
  with beam.Pipeline(options=pipeline_options) as p:
    data_to_fit = (
        p
        | beam.io.ReadFromCsv(INPUT, splittable=False)
        | beam.Map(extract_feature_and_target))

    detector1 = anomaly.AnomalyDetector(algorithm="NaiveBayesian")
    detector2 = anomaly.AnomalyDetector(algorithm="LogisticRegression")

    results = (data_to_fit
               | anomaly.AnomalyDetection(detectors=[detector1, detector2]))

    _ = results | beam.Map(debug_print)

if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  run()
