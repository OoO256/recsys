from abc import ABC, abstractmethod

from typing import Literal, List, Union


class Predictor(ABC):
    def __init__(self, desc=None) -> None:
        self.desc = desc if desc is not None else self.__class__.__name__

    @abstractmethod
    def fit(self, ratings_train, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, ratings_test, *args, **kwargs):
        pass

    def evaluate(
        self,
        input,
        pred,
        metrics: Union[Literal["mae", "rmse"], List[Literal["mae", "rmse"]]] = [
            "mae",
            "rmse",
        ],
    ) -> Union[float, dict]:
        if isinstance(metrics, list):
            results = {}
            for metric in metrics:
                result = self.evaluate(input, pred, metric)
                if isinstance(result, dict):
                    results.update(result)
                else:
                    results[metric] = result

            return results
        else:
            metric = metrics
            if metric == "rmse":
                return ((input["rating"] - pred["rating_pred"]) ** 2).mean() ** 0.5
            elif metric == "mae":
                return (input["rating"] - pred["rating_pred"]).abs().mean()
            else:
                raise ValueError(
                    f"metric should be one of ['mae', 'rmse'], but got {metric}"
                )


class ConstantPredictor(Predictor):
    def __init__(self, desc=None, constant_rating=2.75) -> None:
        super().__init__(desc)
        self.constant_rating = constant_rating

    def fit(self, ratings_train):
        pass

    def predict(self, ratings_test):
        pred = ratings_test
        pred["rating_pred"] = self.constant_rating
        return pred
