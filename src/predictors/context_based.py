import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import Literal, List, Union
from tqdm import tqdm
from functools import lru_cache
from predictors.base import Predictor
from embedding import w2v_average_embedding, sentence_embedding
from sklearn.metrics.pairwise import cosine_similarity
import os
import itertools
import scipy
import random


def normalize_rating(rating):
    # [0.5, 5] => [-1, 1]
    return (rating - 0.5) / 4.5 * 2 - 1


def denormalize_rating(rating):
    # [-1, 1] => [0.5, 5]
    return (rating + 1) / 2 * 4.5 + 0.5


class ContentVectorBased(Predictor):
    def __init__(
        self,
        desc=None,
        vectorizing_method: Literal["tf-idf", "word2vec", "gpt"] = "tf-idf",
        *args,
        **kwargs,
    ) -> None:
        self.vectorizing_method = vectorizing_method
        super().__init__(desc, *args, **kwargs)

    def get_similar_movies(self, movie_title, k=10):
        title_to_index = pd.Series(
            self.movies.index, index=self.movies["title"]
        ).drop_duplicates()
        idx_query = title_to_index[movie_title]
        sim_scores = list(
            enumerate(
                cosine_similarity(
                    self.content_vectors[idx_query].reshape(1, -1), self.content_vectors
                ).squeeze()
            )
        )
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1 : k + 1]
        indices = [i[0] for i in sim_scores]
        return self.movies.iloc[indices]

    def vectorize(self, movies):
        if self.vectorizing_method == "tf-idf":
            tfidf_vectorizer = TfidfVectorizer(stop_words="english")
            return tfidf_vectorizer.fit_transform(movies["overview"].fillna(""))

        elif self.vectorizing_method == "word2vec":
            embedding = w2v_average_embedding(movies["overview"].fillna(""))
            return embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        elif self.vectorizing_method == "gpt":
            embedding = sentence_embedding(movies["overview"].fillna(""))
            return embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        else:
            raise ValueError(
                f"vectorizing_method should be one of ['tf-idf', 'word2vec', 'gpt'], but got {self.vectorizing_method}"
            )

    def fit(self, ratings, movies):
        if os.path.exists(
            f"/home/yonguk/recsys/embeddings/{self.vectorizing_method}.npy"
        ):
            self.content_vectors = np.load(
                f"/home/yonguk/recsys/embeddings/{self.vectorizing_method}.npy",
                allow_pickle=True,
            )

            if self.content_vectors.dtype == np.dtype("O"):
                self.content_vectors = self.content_vectors.item()
        else:
            self.content_vectors = self.vectorize(movies)
            np.save(
                f"/home/yonguk/recsys/embeddings/{self.vectorizing_method}.npy",
                self.content_vectors,
                allow_pickle=True,
            )

        self.movies = movies
        self.ratings_fitted = ratings
        self.ratings_grouped_by_user = ratings.groupby("userId")

    @lru_cache(maxsize=1)
    def get_user_profile(self, user_id):
        if user_id not in self.ratings_grouped_by_user.groups:
            return np.zeros(self.content_vectors.shape[1])

        user_data = self.ratings_grouped_by_user.get_group(user_id)
        valid_ratings = user_data[user_data["movieId"] < self.content_vectors.shape[0]]

        if valid_ratings.empty:
            return np.zeros(self.content_vectors.shape[1])

        if self.vectorizing_method == "tf-idf":
            user_profile = (
                self.content_vectors[valid_ratings["movieId"].values].multiply(
                    normalize_rating(valid_ratings["rating"].values.reshape(-1, 1))
                )
            ).sum(axis=0)
        else:
            user_profile = (
                self.content_vectors[valid_ratings["movieId"].values]
                * normalize_rating(valid_ratings["rating"].values.reshape(-1, 1))
            ).sum(axis=0)

        # l2 normalize
        user_profile = user_profile / np.linalg.norm(user_profile)
        return user_profile

    def predict(self, ratings_test):
        # sort ratings_test by userId
        ratings_test_sorted = ratings_test.sort_values("userId")
        pred = []
        for user_id, movie_id in tqdm(
            zip(ratings_test_sorted["userId"], ratings_test_sorted["movieId"]),
            desc="predicting",
            total=len(ratings_test_sorted),
        ):
            if user_id not in self.ratings_grouped_by_user.groups:
                pred.append(0)
                continue

            if int(movie_id) >= self.content_vectors.shape[0]:
                pred.append(0)
                continue

            user_profile = self.get_user_profile(user_id)
            movie_profile = self.content_vectors[int(movie_id)]
            if self.vectorizing_method == "tf-idf":
                movie_profile = movie_profile.toarray()
            cosine = user_profile.dot(movie_profile.squeeze()).item()
            pred.append(cosine)

        ratings_test_sorted["rating_pred"] = [
            denormalize_rating(cosine) for cosine in pred
        ]
        ratings_test = ratings_test.merge(
            ratings_test_sorted[["userId", "movieId", "rating_pred"]],
            on=["userId", "movieId"],
            how="left",
        )
        return ratings_test

    def evaluate(
        self,
        input,
        pred,
        metrics: Union[Literal["mae", "rmse"], List[Literal["mae", "rmse"]]] = [
            "mae",
            "rmse",
            "ranking_acc",
        ],
    ):
        if isinstance(metrics, list):
            result = super().evaluate(input, pred, metrics)
            # case np array
            result = {
                k: v.item() if isinstance(v, np.ndarray) else v
                for k, v in result.items()
            }
            return result
        elif metrics == "ranking_acc":
            grouped_input = input.groupby("userId")
            acc_per_user = []
            num_total_users = len(grouped_input)
            num_cold_users = 0
            num_one_rating_users = 0
            for user_id, group in tqdm(
                grouped_input, desc="ranking_acc", total=len(grouped_input), leave=False
            ):
                if user_id not in self.ratings_grouped_by_user.groups:
                    num_cold_users += 1

                if len(group) == 1:
                    num_one_rating_users += 1
                    continue

                user_profile = self.get_user_profile(user_id)
                acc_of_user = []
                group = group.sort_values("rating", ascending=False)
                scores = []

                for movie_id in group["movieId"]:
                    if int(movie_id) >= self.content_vectors.shape[0]:
                        movie_profile = np.zeros(self.content_vectors.shape[1])
                    else:
                        movie_profile = self.content_vectors[int(movie_id)]

                    if isinstance(movie_profile, scipy.sparse.csr_matrix):
                        movie_profile = movie_profile.toarray()

                    scores.append(user_profile.dot(movie_profile.squeeze()).item())

                for higher, lower in itertools.combinations(scores, 2):
                    acc_of_user.append(int(higher >= lower))

                if acc_of_user:
                    acc_per_user.append(np.mean(acc_of_user))

            acc = np.mean(acc_per_user).item()
            return {
                "ranking_acc": acc,
                "num_total_users": num_total_users,
                "num_cold_users": num_cold_users,
                "num_one_rating_users": num_one_rating_users,
            }

        else:
            return super().evaluate(input, pred, metrics)
