import argparse
import logging

import apache_beam as beam
from apache_beam.ml.anomaly.aggregations import AverageScore
from apache_beam.ml.anomaly.aggregations import MajorityVote
from apache_beam.ml.anomaly.detectors.sad import StandardAbsoluteDeviation
from apache_beam.ml.anomaly.detectors.mad import MedianAbsoluteDeviation
from apache_beam.ml.anomaly.detectors.loda import Loda
from apache_beam.ml.anomaly.thresholds import FixedThreshold
from apache_beam.ml.anomaly.thresholds import QuantileThreshold
from apache_beam.ml.anomaly.transforms import AnomalyDetection
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions


def run(argv=None, save_main_session=True):
  """
  Args:
    argv: Command line arguments defined for this example.
    save_main_session: Used for internal testing.
  """
  parser = argparse.ArgumentParser()
  _, pipeline_args = parser.parse_known_args(argv)

  # We use the save_main_session option because one or more DoFn's in this
  # workflow rely on global context (e.g., a module imported at module level).
  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

  data = [
      (1, beam.Row(x1=1, x2=4)),
      (2, beam.Row(x1=100, x2=5)),  # an row with a different key (key=2)
      (1, beam.Row(x1=2, x2=4)),
      (1, beam.Row(x1=3, x2=5)),
      (1, beam.Row(x1=10, x2=4)),  # outlier in key=1, with respect to x1
      (1, beam.Row(x1=2, x2=10)),  # outlier in key=1, with respect to x2
      (1, beam.Row(x1=3, x2=4)),
  ]

  detectors = []
  detectors.append(
      StandardAbsoluteDeviation(
          window_size=100,
          features=["x2"],
          threshold_criterion=FixedThreshold(3),
          model_id="sad_x2"))

  detectors.append(
      MedianAbsoluteDeviation(
          window_size=50,
          features=["x1"],
          threshold_criterion=QuantileThreshold(0.99),
          model_id="mad_x1"))

  detectors.append(
      Loda(
          n=3,
          n_init=2,
          features=["x1", "x2"],
          aggregation_strategy=AverageScore(),
          threshold_criterion=FixedThreshold(3),
          model_id="loda"))

  # The pipeline will be run on exiting the with block.
  with beam.Pipeline(options=pipeline_options) as p:
    _ = (
        p | beam.Create(data)
        | beam.Map(lambda t: (t[0], beam.Row(**t[1]._asdict())))
        | AnomalyDetection(detectors, aggregation_strategy=MajorityVote())
        | beam.Map(logging.info))


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()
