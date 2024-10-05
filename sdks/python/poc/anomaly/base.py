"""Base classes for anomaly detection"""

# from typing import Iterable
from typing import Optional
from typing import TypeVar
from typing import List
import uuid

import river.anomaly

import apache_beam as beam
from apache_beam.utils import timestamp
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec
from apache_beam.coders import DillCoder

import river
import river.naive_bayes
from river import compose
from river import linear_model
from river import preprocessing

InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')

__all__ = ['AnomalyDetector', 'AnomalyDetection']


class LogisticRegressionDetector(compose.Pipeline):
  def __init__(self):
    self._model = compose.Pipeline(preprocessing.StandardScaler(),
                                   linear_model.LogisticRegression())

  def predict_one(self, x: dict, **params):
    return self._model.predict_one(x, **params)

  def learn_one(self, x: dict, y=None, **params):
    return self._model.learn_one(x, y, **params)

KNOWN_SUPERVISED_ALGORITHMS= {
  "LogisticRegression": LogisticRegressionDetector,
  "LR": LogisticRegressionDetector,
  "NaiveBayesian": river.naive_bayes.GaussianNB,
  "NB": river.naive_bayes.GaussianNB,
}

KNOWN_UNSUPERVISED_ALGORITHMS = {
  "IncrementalLocalOutlierFactor": river.anomaly.lof.LocalOutlierFactor,
  "iLOF": river.anomaly.lof.LocalOutlierFactor,
  "HalfSpaceTree": river.anomaly.HalfSpaceTrees,
  "HSF": river.anomaly.HalfSpaceTrees,
  "OneClassSVM": river.anomaly.OneClassSVM,
}

KNOWN_ALGORITHMS = KNOWN_SUPERVISED_ALGORITHMS | KNOWN_UNSUPERVISED_ALGORITHMS

class AnomalyDetector:
  def __init__(self, algorithm:Optional[str] = None, label:Optional[str] = None, *args, **kwargs) -> None:
    if algorithm in KNOWN_ALGORITHMS:
      detector = KNOWN_ALGORITHMS[algorithm]
      self._underlying = detector(*args, **kwargs)
      if label:
        self._label = label
      else:
        self._label = f"{algorithm}_{uuid.uuid4().hex}"
      if algorithm in KNOWN_SUPERVISED_ALGORITHMS:
        self._is_supervised = True
      else:
        self._is_supervised = False
    else:
      raise NotImplementedError

  @property
  def label(self):
    return self._label

  @property
  def is_supervised(self):
    return self._is_supervised

  def score_and_learn(self, element):
    if self.is_supervised:
      key, (x, y) = element
      y_pred = self._underlying.predict_one(x)
      self._underlying.learn_one(x, y)
      return key, (x, y, y_pred)
    else:
      key, (x, y) = element
      y_pred = self._underlying.score_one(x)
      self._underlying.learn_one(x)
      return key, (x, y, y_pred)


class ScoreAndLearn(beam.DoFn):
  MODEL_STATE_INDEX = ReadModifyWriteStateSpec('model_state', DillCoder())

  def __init__(self, detector):
    self.detector = detector

  def process(self,
              element,
              model_state=beam.DoFn.StateParam(MODEL_STATE_INDEX),
              **kwargs):

    _, elem = element
    model = model_state.read()
    if model is None:
      model = self.detector

    yield model.score_and_learn(elem)

    model_state.write(model)

class AnomalyDetection(beam.PTransform[beam.PCollection[InputT],
                                       beam.PCollection[OutputT]]):

  def __init__(self, detectors:Optional[List[AnomalyDetector]] = None, agg_strategy = None) -> None:
    self._detectors = detectors

  def maybe_add_key(self, element):
    # TODO: may not need to add keys if there is an existing one?
    key = timestamp.Timestamp.now().micros
    return key, element

  def extract_results(self, element):
    _, result_dict = element
    model_keys = tuple(result_dict.keys())
    assert len(model_keys) > 0

    # x, y are the same across all models
    x = result_dict[model_keys[0]][0][0]
    y = result_dict[model_keys[0]][0][1]
    y_pred = {k: v[0][2] for k, v in result_dict.items()}
    return x, y, y_pred

  def expand(self, input_row: beam.PCollection[InputT]) -> beam.PCollection[OutputT]:
    data = (input_row | "Add key" >> beam.Map(self.maybe_add_key))
    assert self._detectors is not None

    model_results = {}
    for detector in self._detectors:
      model_results[detector.label] = (data
                                       | f"Add dummy key to the input of {detector}" >> beam.WithKeys(1)
                                       | f"Score with {detector}" >> beam.ParDo(ScoreAndLearn(detector)))

    merged = model_results | beam.CoGroupByKey() | "Remove key" >> beam.Map(self.extract_results)

    return merged