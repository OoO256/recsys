from torch.utils.data import Dataset
import pandas as pd
import torch
import lightning.pytorch as pl
from typing import Optional
from utils.split_ratings import split_ratings
import os


class MovieLensDataModule(pl.LightningDataModule):
    def __init__(
        self,
        small: bool = True,
        batch_size: int = 1024,
        test_ratio: float = 0.15,
        val_ratio: Optional[float] = 0.15,
        split_by: str = "random",
        split_kwargs: dict = dict(),
        repeat_train: int = 1,
    ):
        """
        https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data

        movies_metadata.csv: The main Movies Metadata file. Contains information on 45,000 movies featured in the Full MovieLens dataset. Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies.

        keywords.csv: Contains the movie plot keywords for our MovieLens movies. Available in the form of a stringified JSON Object.

        credits.csv: Consists of Cast and Crew Information for all our movies. Available in the form of a stringified JSON Object.

        links.csv: The file that contains the TMDB and IMDB IDs of all the movies featured in the Full MovieLens dataset.

        links_small.csv: Contains the TMDB and IMDB IDs of a small subset of 9,000 movies of the Full Dataset.

        ratings_small.csv: The subset of 100,000 ratings from 700 users on 9,000 movies.
        """
        super().__init__()
        self.dirname = os.path.join(
            os.environ.get("WORKSPACE_FOLDER"), "/datasets/movielens/kaggle"
        )
        self.small = small
        self._read_csv()
        self.max_user_id = self.ratings_df["userId"].max()
        self.max_movie_id = self.ratings_df["movieId"].max()
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.split_by = split_by
        self.split_kwargs = split_kwargs
        self.repeat_train = repeat_train

    def _read_csv(self):
        self.movies_df = pd.read_csv(self.dirname + "/movies_metadata.csv")
        self.keywords_df = pd.read_csv(self.dirname + "/keywords.csv")
        # self.credits_df = pd.read_csv(self.dirname + "/credits.csv")
        if self.small:
            self.links_df = pd.read_csv(self.dirname + "/links_small.csv")
            self.ratings_df = pd.read_csv(self.dirname + "/ratings_small.csv")
        else:
            self.links_df = pd.read_csv(self.dirname + "/links.csv")
            self.ratings_df = pd.read_csv(self.dirname + "/ratings.csv")

        # convert ids to int
        self.ratings_df["userId"] = self.ratings_df["userId"].astype(int)
        self.ratings_df["movieId"] = self.ratings_df["movieId"].astype(int)

    def setup(self, stage: str) -> None:
        self.train, self.test = split_ratings(
            self.ratings_df,
            split_by=self.split_by,
            test_ratio=self.test_ratio,
            **self.split_kwargs
        )
        self.train, self.val = split_ratings(
            self.train,
            split_by=self.split_by,
            test_ratio=self.val_ratio,
            **self.split_kwargs
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            MovieLensDataset(self.train, repeat=self.repeat_train),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            MovieLensDataset(self.val),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            MovieLensDataset(self.test),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=False,
        )


class MovieLensDataset(Dataset):
    def __init__(self, ratings_df, repeat: int = 1):
        super().__init__()
        self.ratings_df = ratings_df
        self.repeat = repeat
        self.user_ids_tensor = torch.from_numpy(self.ratings_df["userId"].values)
        self.movie_ids_tensor = torch.from_numpy(self.ratings_df["movieId"].values)
        self.ratings_tensor = torch.from_numpy(self.ratings_df["rating"].values)

    def __len__(self):
        return len(self.ratings_df) * self.repeat

    def __getitem__(self, idx):
        idx = idx % len(self.ratings_tensor)
        user_id = self.user_ids_tensor[idx]
        movie_id = self.movie_ids_tensor[idx]
        rating = self.ratings_tensor[idx]
        return {
            "userId": user_id,
            "movieId": movie_id,
            "rating": rating,
        }
