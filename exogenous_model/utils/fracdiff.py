import numpy as np
import pandas as pd

class FracDifferentiator:
    def __init__(self, d=0.7, thresh=1e-5):
        self.d = d
        self.thresh = thresh
        self.weights = None
        self.width = None

    def _compute_weights(self, size):
        w = [1.]
        for k in range(1, size):
            w_ = -w[-1] * (self.d - k + 1) / k
            if abs(w_) < self.thresh:
                break
            w.append(w_)
        self.weights = np.array(w[::-1])
        self.width = len(self.weights)

    def transform(self, series: pd.Series) -> pd.Series:
        if self.weights is None or self.width is None:
            self._compute_weights(len(series))

        if len(series) < self.width:
            raise ValueError(f"Not enough data for fractional differencing. Required: {self.width}, got: {len(series)}.")

        diff_series = []
        for i in range(self.width, len(series)):
            window = series.iloc[i - self.width:i]
            if window.isnull().any():
                diff_series.append(np.nan)
            else:
                diff_series.append(np.dot(self.weights, window))

        return pd.Series([np.nan] * self.width + diff_series, index=series.index)
