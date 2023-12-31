from predictors.base import Predictor


class AverageRatingPredictor(Predictor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def fit(self, ratings_train, moives, *args, **kwargs):
        self.avg_rating = ratings_train.groupby("movieId")["rating"].mean()

    def predict(self, ratings_test):
        return ratings_test.merge(
            self.avg_rating, on="movieId", how="left", suffixes=("", "_pred")
        )


class WeightedRatingPredictor(Predictor):
    """
    https://math.stackexchange.com/questions/169032/understanding-the-imdb-weighted-rating-function-for-usage-on-my-own-website
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def fit(self, ratings_train, moives, *args, **kwargs):
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

        movies["rating_pred"] = movies.apply(weighted_rating, axis=1)
        self.weighted_rating = movies

    def predict(self, ratings_test):
        return ratings_test.merge(self.weighted_rating, on="movieId", how="left")
