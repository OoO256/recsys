from typing import Literal
import pandas as pd


def split_ratings(
    ratings,
    split_by: Literal["random", "user", "time"],
    test_ratio: float,
    **kwargs,
):
    if split_by == "random":
        return split_ratings_by_random(ratings, test_ratio, **kwargs)
    elif split_by == "user":
        return split_ratings_by_user(ratings, test_ratio, **kwargs)
    elif split_by == "time":
        return split_ratings_by_time(ratings, test_ratio, **kwargs)
    else:
        raise ValueError(
            f"split_method should be one of ['random', 'user', 'time'], but got {split_by}"
        )


def split_ratings_by_random(ratings, test_ratio: float):
    test = ratings.sample(frac=test_ratio)
    train = ratings.drop(test.index)
    return train, test


def split_ratings_by_time(ratings, test_ratio: float):
    threshold = ratings["timestamp"].quantile(1 - test_ratio)
    train = ratings[ratings["timestamp"] <= threshold]
    test = ratings[ratings["timestamp"] > threshold]
    return train, test


def split_ratings_by_user(
    ratings, test_ratio: float, min_ratings: int = 5, warm_only: bool = False
):
    if warm_only:
        user_counts = ratings[["userId", "movieId"]]
        user_counts = user_counts.groupby("userId").count()
        user_counts = user_counts.reset_index()
        warm_users = user_counts[user_counts["movieId"] >= min_ratings]["userId"].values
        warm_ratings = ratings[ratings["userId"].isin(warm_users)]
        train_ratings = warm_ratings.sample(frac=1 - test_ratio)
        test_ratings_temp = warm_ratings.drop(train_ratings.index)
        test_ratings_temp["ratingIndex"] = test_ratings_temp.groupby(
            "userId"
        ).cumcount()
        test_ratings = test_ratings_temp[
            test_ratings_temp["ratingIndex"] >= min_ratings
        ]
        cold_test_ratings = test_ratings_temp[
            test_ratings_temp["ratingIndex"] < min_ratings
        ]
        cold_test_ratings = cold_test_ratings.drop(columns=["ratingIndex"])
        train_ratings = pd.concat([train_ratings, cold_test_ratings])
        return train_ratings, test_ratings
    else:
        user_counts = ratings[["userId", "movieId"]]
        user_counts = user_counts.groupby("userId").count()
        users_test = user_counts.sample(frac=test_ratio)
        ratings_test = ratings[ratings["userId"].isin(users_test.index)]
        ratings_train = ratings.drop(ratings_test.index)
        return ratings_train, ratings_test
