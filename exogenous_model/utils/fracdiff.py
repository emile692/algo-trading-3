# exogenous_model/dataset/frac_differentiator.py
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

class FracDifferentiator:
    def __init__(self, d=None, d_values=None, thresh=1e-5):
        """
        Si d est donné, on suppose qu'on est en mode inférence (on ne cherche pas d optimal)
        Sinon, on peut chercher le d optimal à partir d'une liste de d_values
        """
        self.d = d
        self.d_values = d_values or np.arange(0.1, 1.0, 0.1)
        self.thresh = thresh
        self.weights = None
        self.width = None

    def _compute_weights(self, d, size):
        w = [1.]
        for k in range(1, size):
            w_ = -w[-1] * (d - k + 1) / k
            if abs(w_) < self.thresh:
                break
            w.append(w_)
        return np.array(w[::-1])

    def _frac_diff(self, series, weights):
        width = len(weights)
        diff_series = []
        for i in range(width, len(series)):
            window = series.iloc[i - width:i]
            if window.isnull().any():
                diff_series.append(np.nan)
            else:
                diff_series.append(np.dot(weights, window))
        return pd.Series([np.nan] * width + diff_series, index=series.index)

    def find_optimal_d(self, series: pd.Series, pvalue_threshold=0.05):
        """
        Calcule et choisit le meilleur d pour rendre la série stationnaire (ADF test)
        """
        for d in self.d_values:
            weights = self._compute_weights(d, len(series))
            transformed = self._frac_diff(series, weights).dropna()
            if len(transformed) < 20:
                continue
            pval = adfuller(transformed, maxlag=1, regression='c', autolag=None)[1]
            if pval < pvalue_threshold:
                self.d = d
                self.weights = weights
                self.width = len(weights)
                return d, pval
        # Par défaut, si rien n'est stationnaire
        self.d = self.d_values[0]
        self.weights = self._compute_weights(self.d, len(series))
        self.width = len(self.weights)
        return self.d, None

    def fit(self, series: pd.Series):
        """
        A appeler à l'entraînement (cherche d optimal)
        """
        return self.find_optimal_d(series)

    def transform(self, series: pd.Series) -> pd.Series:
        """
        Applique la frac diff avec d et les poids existants (en inférence ou après fit)
        """
        if self.weights is None:
            if self.d is None:
                raise ValueError("You must fit or set d before calling transform.")
            self.weights = self._compute_weights(self.d, len(series))
            self.width = len(self.weights)

        if len(series) < self.width:
            raise ValueError(f"Not enough data for fractional differencing. Required: {self.width}, got: {len(series)}.")

        return self._frac_diff(series, self.weights)
