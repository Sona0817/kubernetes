import bentoml
import numpy as np
from bentoml.io import NumpyNdarray, PandasDataFrame
import pandas as pd



# SURFACE_CLASSES = ['negatieve', 'positive']
#
# runner = bentoml.sklearn.get("adaboost:latest").to_runner()
#
# svc = bentoml.Service("adaboost-svc", runners=[runner])
#
# @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
# def classify(input_series):
#     # inference preprocess
#
#     result = runner.predict.run(input_series)
#     return np.array([SURFACE_CLASSES[i] for i in result])
#

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

@svc.api(input=PandasDataFrame(), output=NumpyNdarray())
def predict(input_series: pd.DataFrame) -> np.ndarray:
    result = iris_clf_runner.predict.run(input_series)
    return result
