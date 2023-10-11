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
from predictors.collaborative import NMF, GeneralMatrixFactorizer, MultiLayerPerceptron
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
        [False],
        [
            ("random", {}),
            # ("time", {}),
            # ("user", {"warm_only": False}),
            # ("user", {"warm_only": True, "min_ratings": 5}),
            # ("user", {"warm_only": True, "min_ratings": 20}),
        ],
    ]
    result = {
        "Dataset": [],
    }

    for small, (split_by, split_kwargs) in itertools.product(*somelists):
        dataset = MovieLensDataModule(
            small=small,
            split_by=split_by,
            split_kwargs=split_kwargs,
            batch_size=1024 if small else 8 * 1024,
        )
        train, test = split_ratings(dataset.ratings_df, split_by, 0.15, **split_kwargs)

        result["Dataset"].append(
            f"{'small' if small else 'large'}, split_by {split_by}, {split_kwargs}"
        )

        # benchmark demographic predictors
        for predictor in [AverageRatingPredictor(), WeightedRatingPredictor()]:
            predictor.fit(train)
            pred = predictor.predict(test)
            eval = predictor.evaluate(test, pred)

            result[predictor.__class__.__name__] = result.get(
                predictor.__class__.__name__, []
            )
            result[predictor.__class__.__name__].append(round_dict(eval))
            print(f"{predictor.__class__.__name__} eval: {round_dict(eval)}")

        # benchmark collaborative predictors
        for module in [
            GeneralMatrixFactorizer(
                dataset.max_user_id + 1, dataset.max_movie_id + 1, learning_rate=0.01
            ),
            MultiLayerPerceptron(
                dataset.max_user_id + 1, dataset.max_movie_id + 1, learning_rate=0.01
            ),
            NMF(dataset.max_user_id + 1, dataset.max_movie_id + 1, learning_rate=0.01),
        ]:
            wandb_logger = WandbLogger(
                f"benchmark/{module.__class__.__name__}/{'small' if small else 'large'}_{split_by}_{split_kwargs}",
                project="recsys",
                log_model=True,
            )
            trainer = pl.Trainer(
                max_epochs=100,
                callbacks=[
                    EarlyStopping(monitor="val/loss", patience=3, min_delta=0.001),
                    LearningRateMonitor(logging_interval="step"),
                ],
                logger=wandb_logger,
            )
            trainer.fit(module, datamodule=dataset)
            eval = trainer.test(module, datamodule=dataset)[-1]

            result[module.__class__.__name__] = result.get(
                module.__class__.__name__, []
            )
            result[module.__class__.__name__].append(round_dict(eval))
            print(f"{module.__class__.__name__} eval: {round_dict(eval)}")

            gc.collect()
            torch.cuda.empty_cache()

        # print and save result
        result_str = tabulate_result(result)
        print(result_str)
        with open(BECHMARK_DFAULT_PATH, "w") as f:
            f.write(result_str)

        # garbage collection
        del dataset
        del train
        del test
        sleep(10)

    result_str = tabulate_result(result)
    print(result_str)
    with open(BECHMARK_DFAULT_PATH, "w") as f:
        f.write(result_str)

    wandb.finish()

    return result, result_str


if __name__ == "__main__":
    run_benchmark()
