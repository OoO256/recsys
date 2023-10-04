import lightning.pytorch as pl
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SimpleMatrixFactorizer(pl.LightningModule):
    def __init__(self, num_users, num_items, latent_dim=256, learning_rate=0.001):
        super(SimpleMatrixFactorizer, self).__init__()
        self.save_hyperparameters()

        self.user_embedding = torch.nn.Embedding(num_users, latent_dim)
        self.item_embedding = torch.nn.Embedding(num_items, latent_dim)
        self.user_bais = torch.nn.Parameter(torch.zeros(latent_dim))
        self.item_bais = torch.nn.Parameter(torch.zeros(latent_dim))
        self.bais = torch.nn.Parameter(torch.zeros(1))
        self.learning_rate = learning_rate

        self.optimizer = Adam(self.parameters(), lr=self.learning_rate)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, patience=5)

    def forward(self, user_indices, item_indices):
        user_vecs = self.user_embedding(user_indices) + self.user_bais
        item_vecs = self.item_embedding(item_indices) + self.item_bais
        return torch.sum(user_vecs * item_vecs, dim=1) + self.bais

    def calculate_loss_and_metrics(self, batch, prefix=""):
        user_indices = batch["userId"]
        item_indices = batch["movieId"]
        ratings = batch["rating"]
        preds = self.forward(user_indices, item_indices)
        mae = torch.nn.functional.l1_loss(preds, ratings)
        rmse = torch.sqrt(torch.nn.functional.mse_loss(preds, ratings))
        loss = mae
        lr = self.optimizer.param_groups[0]["lr"]
        return loss, {
            prefix + "mae": mae,
            prefix + "rmse": rmse,
            prefix + "lr": lr,
        }

    def training_step(self, batch, batch_idx):
        loss, metrics = self.calculate_loss_and_metrics(batch, prefix="train/")
        self.log_dict(
            {"train/loss": loss, **metrics},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.calculate_loss_and_metrics(batch, prefix="val/")
        self.log_dict(
            {"val/loss": loss, **metrics},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, metrics = self.calculate_loss_and_metrics(batch, prefix="test/")
        self.log_dict(
            {"test/loss": loss, **metrics},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "monitor": "val/loss",
            },
        }
