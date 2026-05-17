import numpy as np
from .funcs import *
from sklearn.base import BaseEstimator


class URLFNoise(BaseEstimator):
    def __init__(self, noise_level=0.1, random_state=42):
        self.noise_level = noise_level
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform_y(self, y):
        return urlf(
            y,
            noise_level=self.noise_level,
            random_state=self.random_state
        )

    def fit_resample(self, X, y):
        return X, self.transform_y(y)

class NARNoise(BaseEstimator):
    def __init__(
        self,
        noise_levels=None,
        random_state=42,
        random_range=(0.0, 0.2)
    ):
        self.noise_levels = noise_levels
        self.random_state = random_state
        self.random_range = random_range

    def fit(self, X, y=None):
        return self

    def transform_y(self, y):
        return nar(
            y,
            noise_levels=self.noise_levels,
            random_state=self.random_state,
            random_range=self.random_range
        )

    def fit_resample(self, X, y):
        return X, self.transform_y(y)