import numpy as np


def fit_function(predict, mos):
    params = np.polyfit(predict, mos, deg=3)
    p1 = np.poly1d(params)
    predict_fitted = p1(predict)
    return predict_fitted
