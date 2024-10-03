import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.utils import timestamp

from river import compose
from river import linear_model
from river import metrics
from river import preprocessing
from river import naive_bayes

INPUT = "/Users/shunping/Projects/beam-dev-python-anomaly/sdks/python/poc/phishing.csv"

model1 = compose.Pipeline(preprocessing.StandardScaler(),
                          linear_model.LogisticRegression())
model2 = naive_bayes.GaussianNB()

metric1 = metrics.Accuracy()
metric2 = metrics.Accuracy()


def extract_feature_and_target(element):
  data = element._asdict()
  target = 'is_phishing'
  x = {k: v for k, v in data.items() if k != target}
  y = bool(data[target])
  key = timestamp.Timestamp.now().micros
  return key, (x, y)

def run_logistic_regression(element):
  model = model1
  key, (x, y) = element
  y_pred = model.predict_one(x)
  metric1.update(y, y_pred)
  model.learn_one(x, y)
  return key, (x, y, y_pred)

def run_nb(element):
  model = model2
  key, (x, y) = element
  y_pred = model.predict_one(x)
  metric2.update(y, y_pred)
  model.learn_one(x, y)
  return key, (x, y, y_pred)

def collect_metrics(element):
  x, y, y_pred = element
  metric1.update(y, y_pred['model1'])
  metric2.update(y, y_pred['model2'])
  print("metric1", metric1)
  print("metric2", metric2)

def debug_print(element):
  print(element)

def extract_result(element):
  _, result_dict = element
  model_keys = tuple(result_dict.keys())
  assert len(model_keys) > 0

  # x, y are the same across all models
  x = result_dict[model_keys[0]][0][0]
  y = result_dict[model_keys[0]][0][1]
  y_pred = {k: v[0][2] for k, v in result_dict.items()}
  return x, y, y_pred

def run():
  pipeline_options = PipelineOptions(None)
  with beam.Pipeline(options=pipeline_options) as p:
    data_to_fit = (
        p
        | beam.io.ReadFromCsv(INPUT, splittable=False)
        | beam.Map(extract_feature_and_target))

    model1_result = (data_to_fit | beam.Map(run_logistic_regression))

    model2_result = (data_to_fit | beam.Map(run_nb))

    merge = ({
        'model1': model1_result,
        'model2': model2_result
    }
             | beam.CoGroupByKey())
    _ = merge | beam.Map(extract_result) | beam.Map(collect_metrics)

if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  run()
