
from typing import Optional

from apache_beam.ml.anomaly.base import BaseAnomalyModel
from apache_beam.ml.anomaly.models.sad import StandardAbsoluteDeviation
from apache_beam.ml.anomaly.models.mad import MedianAbsoluteDeviation
from apache_beam.ml.anomaly.models.loda import LodaWeakLearner
from apache_beam.ml.anomaly.models.river import LocalOutlierFactor

KNOWN_ALGORITHMS: dict[str, Optional[type[BaseAnomalyModel]]] = {
    "sad": StandardAbsoluteDeviation,
    "mad": MedianAbsoluteDeviation,
    "loda": LodaWeakLearner,
    "ilof": LocalOutlierFactor,
    # # "HSF": river.anomaly.HalfSpaceTrees,
    # # "OneClassSVM": river.anomaly.OneClassSVM,
    "ensemble": None,  # umbrella for all ensemble algorithms
}
