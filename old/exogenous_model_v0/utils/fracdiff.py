# exogenous_model_v0/dataset/frac_differentiator.py
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

    def get_ffd_weights(self, d):
        """Calcule les poids pour la différenciation fractionnaire à fenêtre fixe"""
        weights = [1.0]
        k = 1
        while True:
            weight = -weights[-1] * (d - k + 1) / k
            if abs(weight) < self.thresh:
                break
            weights.append(weight)
            k += 1
        return np.array(weights[::-1])

    def frac_diff_ffd(self, series, d):
        """Différenciation fractionnaire avec fenêtre fixe (FFD)"""
        weights = self.get_ffd_weights(d)
        window_size = len(weights)
        diff_series = []

        for i in range(window_size, len(series)):
            window = series.iloc[i - window_size:i]
            if window.isnull().any():
                diff_series.append(np.nan)
            else:
                diff_value = np.dot(weights, window)
                diff_series.append(diff_value)

        return pd.Series(diff_series, index=series.index[window_size:])

    def find_optimal_d(self, series: pd.Series, pvalue_threshold: float = 0.05, max_weights=1000) -> tuple:
        """
        Retourne le plus petit `d` qui rend la série stationnaire.
        Limite la taille de la fenêtre des poids.
        """
        for d in sorted(self.d_values):  # d croissants, du plus petit au plus grand
            weights = self.get_ffd_weights(d)
            if len(weights) > max_weights:
                continue  # ignore d trop petit avec trop de poids

            transformed = self.frac_diff_ffd(series, d).dropna()
            if len(transformed) < 20:
                continue

            pval = adfuller(transformed, regression='c', autolag='AIC')[1]

            if pval < pvalue_threshold:
                self.d = d
                self.weights = weights
                self.width = len(weights)
                return d, pval

        # fallback
        d = self.d_values[-1]
        weights = self.get_ffd_weights(d)
        self.d = d
        self.weights = weights
        self.width = len(weights)
        return d, None

    def fit(self, series: pd.Series):
        """
        A appeler à l'entraînement (cherche d optimal)
        """
        return self.find_optimal_d(series)

    def transform(self, series: pd.Series) -> pd.Series:
        """
        Applique la frac diff avec d et les poids existants (en inférence ou après fit)
        Utilise la méthode FFD
        """
        if self.d is None:
            raise ValueError("You must fit or set d before calling transform.")

        if self.weights is None:
            self.weights = self.get_ffd_weights(self.d)
            self.width = len(self.weights)

        window_size = len(self.weights)
        diff_series = []

        for i in range(window_size, len(series)):
            window = series.iloc[i - window_size:i]
            if window.isnull().any():
                diff_series.append(np.nan)
            else:
                diff_series.append(np.dot(self.weights, window))

        return pd.Series(diff_series, index=series.index[window_size:])