import gc
import itertools
from time import sleep

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from tabulate import tabulate

from datasets.movie import MovieLensDataModule
from predictors.base import ConstantPredictor
from predictors.collaborative import (
    NeuralMatrixFactorizer,
    GeneralMatrixFactorizer,
    MultiLayerPerceptron,
)
from predictors.context_based import ContentVectorBased
from predictors.demographic import AverageRatingPredictor, WeightedRatingPredictor
from utils.split_ratings import split_ratings
import wandb

BECHMARK_DFAULT_PATH = "/home/yonguk/recsys/benchmarks/benchmark.txt"


def tabulate_result(d: dict):
    return tabulate(pd.DataFrame(d), headers="keys", tablefmt="grid")


def round_dict(d: dict, n: int = 3):
    return {k: round(v, n) for k, v in d.items()}


def run_benchmark(path_benchmark=BECHMARK_DFAULT_PATH):
    somelists = [
        [True, False],
        [
            ("random", {}),
            ("time", {}),
            ("user", {"warm_only": False}),
            ("user", {"warm_only": True, "min_ratings": 5}),
            ("user", {"warm_only": True, "min_ratings": 20}),
        ],
    ]
    result = {
        "Dataset": [],
    }

    for small, (split_by, split_kwargs) in itertools.product(*somelists):
        data_module = MovieLensDataModule(
            small=small,
            split_by=split_by,
            split_kwargs=split_kwargs,
            batch_size=1024 if small else 8 * 1024,
        )
        data_module.setup("fit")

        result["Dataset"].append(
            f"{'small' if small else 'large'}, split_by {split_by}, {split_kwargs}"
        )

        # benchmark demographic predictors
        for predictor in [
            AverageRatingPredictor(),
            WeightedRatingPredictor(),
            ConstantPredictor(),
        ]:
            predictor.fit(data_module.train)
            pred = predictor.predict(data_module.test)
            eval = predictor.evaluate(data_module.test, pred)

            result[predictor.desc] = result.get(predictor.desc, [])
            result[predictor.desc].append(round_dict(eval))
            print(f"{predictor.desc} eval: {round_dict(eval)}")

        # content based
        for predictor in [
            ContentVectorBased("tf-idf", "tf-idf"),
            ContentVectorBased("word2vec", "word2vec"),
            ContentVectorBased("gpt", "gpt"),
        ]:
            predictor.fit(data_module.train, data_module.movies_df)
            pred = predictor.predict(data_module.test)
            eval = predictor.evaluate(data_module.test, pred)

            result[predictor.desc] = result.get(predictor.desc, [])
            result[predictor.desc].append(round_dict(eval))
            print(f"{predictor.desc} eval: {round_dict(eval)}")

            sm = predictor.get_similar_movies("The Dark Knight Rises")
            print(sm[["title", "overview", "vote_average"]])

        # benchmark collaborative predictors
        for module in [
            GeneralMatrixFactorizer(
                dataset.max_user_id + 1, dataset.max_movie_id + 1, learning_rate=0.01
            ),
            MultiLayerPerceptron(
                dataset.max_user_id + 1, dataset.max_movie_id + 1, learning_rate=0.01
            ),
            NeuralMatrixFactorizer(
                dataset.max_user_id + 1, dataset.max_movie_id + 1, learning_rate=0.01
            ),
        ]:
            wandb.finish()
            wandb_logger = WandbLogger(
                f"benchmark/{module.__class__.__name__}/{'small' if small else 'large'}_{split_by}_{split_kwargs}",
                project="recsys",
                log_model=True,
                reinit=True,
            )
            trainer = pl.Trainer(
                max_epochs=100,
                callbacks=[
                    EarlyStopping(monitor="val/loss", patience=3, min_delta=0.001),
                    LearningRateMonitor(logging_interval="step"),
                ],
                logger=wandb_logger,
            )
            trainer.fit(module, datamodule=data_module)
            eval = trainer.test(module, datamodule=data_module)[-1]

            result[module.__class__.__name__] = result.get(
                module.__class__.__name__, []
            )
            result[module.__class__.__name__].append(round_dict(eval))
            print(f"{module.__class__.__name__} eval: {round_dict(eval)}")

            del wandb_logger
            gc.collect()
            torch.cuda.empty_cache()

        # print and save result
        result_str = tabulate_result(result)
        print(result_str)
        with open(BECHMARK_DFAULT_PATH, "w") as f:
            f.write(result_str)

        # garbage collection
        del data_module
        sleep(10)

    result_str = tabulate_result(result)
    print(result_str)
    with open(BECHMARK_DFAULT_PATH, "w") as f:
        f.write(result_str)

    wandb.finish()

    return result, result_str


if __name__ == "__main__":
    run_benchmark()
