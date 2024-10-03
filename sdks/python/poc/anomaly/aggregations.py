import collections
import itertools
import math
import statistics
from typing import Iterable

from poc.anomaly.base import AnomalyDecision


def majority_vote(outlier_label=1, normal_label=0, tie_breaker=0, include_history=False):

  def get_vote(decisions: Iterable[AnomalyDecision]) -> AnomalyDecision:
    predictions = itertools.filterfalse(
        lambda prediction: prediction is not None,
        map(lambda decision: decision.prediction, decisions))

    counters = collections.Counter(predictions)
    if len(counters) == 0:
      return AnomalyDecision(model="aggregate", prediction=None)

    if counters[normal_label] < counters[outlier_label]:
      vote = outlier_label
    elif counters[normal_label] > counters[outlier_label]:
      vote = normal_label
    else:
      vote = tie_breaker
    info=('[' + ('; '.join(map(str, decisions))) + ']') if include_history else ''

    return AnomalyDecision(
        model="aggregate",
        prediction=vote,
        info=info)

  return get_vote


# AND scheme
def all_vote(outlier_label=1, normal_label=0, include_history=False):

  def get_vote(decisions: Iterable[AnomalyDecision]) -> AnomalyDecision:
    predictions = itertools.filterfalse(
        lambda prediction: prediction is None,
        map(lambda decision: decision.prediction, decisions))

    vote = outlier_label if all(map(lambda p: p == outlier_label,
                                    predictions)) else normal_label
    info=('[' + ('; '.join(map(str, decisions))) + ']') if include_history else ''

    return AnomalyDecision(
        model="aggregate",
        prediction=vote,
        info=info)

  return get_vote


# OR scheme
def any_vote(outlier_label=1, normal_label=0, include_history=False):

  def get_vote(decisions: Iterable[AnomalyDecision]):
    predictions = list(
        itertools.filterfalse(
            lambda prediction: prediction is None,
            map(lambda decision: decision.prediction, decisions)))

    vote = outlier_label if any(map(lambda p: p == outlier_label,
                                    predictions)) else normal_label
    info=('[' + ('; '.join(map(str, decisions))) + ']') if include_history else ''

    return AnomalyDecision(
        model="aggregate",
        prediction=vote,
        info=info)

  return get_vote

def average_score(include_history=False):
  def get_score(decisions: Iterable[AnomalyDecision]) -> AnomalyDecision:
    scores: Iterable[float] = itertools.filterfalse(
        lambda score: math.isnan(score),
        map(lambda decision: decision.score, decisions))
    info=('[' + ('; '.join(map(str, decisions))) + ']') if include_history else ''

    return AnomalyDecision(model="aggregate", score=statistics.mean(scores), info=info)

  return get_score
