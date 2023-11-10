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
from predictors.base import ConstantPredictor, Predictor
from lightning_modules.collaborative import (
    NeuralMatrixFactorizer,
    GeneralMatrixFactorizer,
    MultiLayerPerceptron,
    BaseLightningModule,
)
from predictors.context_based import ContentVectorBased
from predictors.demographic import AverageRatingPredictor, WeightedRatingPredictor
import wandb

BECHMARK_DFAULT_PATH = "/home/yonguk/recsys/benchmarks/benchmark.txt"


def tabulate_result(d: dict):
    return tabulate(pd.DataFrame(d), headers="keys", tablefmt="grid")


def round_dict(d: dict, n: int = 3):
    return {k: round(v, n) for k, v in d.items()}


def run_benchmark(path_benchmark=BECHMARK_DFAULT_PATH):
    somelists = [
        [True],
        [
            ("random", {}),
            # ("time", {}),
            # ("user", {"warm_only": False}),
            # ("user", {"warm_only": True, "min_ratings": 5}),
            # ("user", {"warm_only": True, "min_ratings": 20}),
        ],
    ]
    benchmark_table = {
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
        desc_dataset = (
            f"{'small' if small else 'large'}, split_by {split_by}, {split_kwargs}"
        )

        benchmark_table["Dataset"].append(desc_dataset)

        # benchmark demographic predictors
        for method in [
            AverageRatingPredictor(),
            WeightedRatingPredictor(),
            ConstantPredictor(),
            ContentVectorBased("tf-idf", "tf-idf"),
            ContentVectorBased("word2vec", "word2vec"),
            ContentVectorBased("gpt", "gpt"),
            GeneralMatrixFactorizer(
                data_module.max_user_id + 1, data_module.max_movie_id + 1
            ),
            MultiLayerPerceptron(
                data_module.max_user_id + 1, data_module.max_movie_id + 1
            ),
            NeuralMatrixFactorizer(
                data_module.max_user_id + 1, data_module.max_movie_id + 1
            ),
        ]:
            if isinstance(method, Predictor):
                method.fit(data_module.train, data_module.movies_df)
                pred = method.predict(data_module.test)
                result = method.evaluate(data_module.test, pred)
            elif isinstance(method, BaseLightningModule):
                wandb.finish()
                wandb_logger = WandbLogger(
                    f"benchmark/{method.__class__.__name__}/{desc_dataset}",
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
                    # logger=wandb_logger,
                )
                trainer.fit(method, datamodule=data_module)
                result = trainer.test(method, datamodule=data_module)[-1]

                del wandb_logger
                gc.collect()
                torch.cuda.empty_cache()

            else:
                raise ValueError(f"predictor should be Predictor or LightningModule")

            benchmark_table[method.desc] = benchmark_table.get(method.desc, [])
            benchmark_table[method.desc].append(round_dict(result))
            print(f"{method.desc} eval: {round_dict(result)}")

            if isinstance(method, ContentVectorBased):
                sm = method.get_similar_movies("The Dark Knight Rises")
                print(sm[["title", "overview", "vote_average"]])

        # print and save result
        result_str = tabulate_result(benchmark_table)
        print(result_str)
        with open(BECHMARK_DFAULT_PATH, "w") as f:
            f.write(result_str)

        # garbage collection
        del data_module
        sleep(10)

    wandb.finish()

    return benchmark_table, result_str


if __name__ == "__main__":
    run_benchmark()
