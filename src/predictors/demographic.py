from predictors.base import Predictor

from typing import Literal, List, Union


class AverageRatingPredictor(Predictor):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, ratings_train):
        self.avg_rating = ratings_train.groupby("movieId")["rating"].mean()

    def predict(self, ratings_test):
        return ratings_test.merge(
            self.avg_rating, on="movieId", how="left", suffixes=("", "_pred")
        )

    def evaluate(
        self,
        input,
        pred,
        metrics: Union[Literal["mae", "rmse"], List[Literal["mae", "rmse"]]] = [
            "mae",
            "rmse",
        ],
    ):
        if isinstance(metrics, list):
            return {metric: self.evaluate(input, pred, metric) for metric in metrics}
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


class WeightedRatingPredictor(Predictor):
    """
    https://math.stackexchange.com/questions/169032/understanding-the-imdb-weighted-rating-function-for-usage-on-my-own-website
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, ratings_train):
        rating_avg = ratings_train["rating"].mean()
        rating_counts = ratings_train.groupby("movieId")["rating"].count()
        min_rate_count = rating_counts.quantile(0.9)

        movies = ratings_train.groupby("movieId")[["rating"]].mean()
        movies["rating_count"] = rating_counts

        def weighted_rating(x):
            vote_count = x["rating_count"]
            vote_average = x["rating"]
            return (vote_count / (vote_count + min_rate_count) * vote_average) + (
                min_rate_count / (min_rate_count + vote_count) * rating_avg
            )

        movies["weighted_rating"] = movies.apply(weighted_rating, axis=1)
        self.weighted_rating = movies

    def predict(self, ratings_test):
        return ratings_test.merge(self.weighted_rating, on="movieId", how="left")

    def evaluate(
        self,
        input,
        pred,
        metrics: Union[Literal["mae", "rmse"], List[Literal["mae", "rmse"]]] = [
            "mae",
            "rmse",
        ],
    ):
        if isinstance(metrics, list):
            return {metric: self.evaluate(input, pred, metric) for metric in metrics}
        else:
            metric = metrics
            if metric == "rmse":
                return ((input["rating"] - pred["weighted_rating"]) ** 2).mean() ** 0.5
            elif metric == "mae":
                return (input["rating"] - pred["weighted_rating"]).abs().mean()
            else:
                raise ValueError(
                    f"metric should be one of ['mae', 'rmse'], but got {metric}"
                )
