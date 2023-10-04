from abc import ABC, abstractmethod


class Predictor(ABC):
    @abstractmethod
    def fit(self, ratings_train):
        pass

    @abstractmethod
    def predict(self, ratings_test):
        pass
