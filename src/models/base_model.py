from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """Abstract base class for all models"""

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_params(self):
        pass

    @abstractmethod
    def self_params(self, **params):
        pass