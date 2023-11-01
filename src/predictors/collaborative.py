import lightning.pytorch as pl
import torch
from torch.optim import Adam
from torch import nn


class BaseCollaborativeFilter(pl.LightningModule):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=16,
        learning_rate=0.01,
    ):
        super(BaseCollaborativeFilter, self).__init__()
        self.save_hyperparameters()

        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)

        self.learning_rate = learning_rate

    def forward(self, user_indices, item_indices):
        raise NotImplementedError()

    def calculate_loss_and_metrics(self, batch, prefix=""):
        user_indices = batch["userId"]
        item_indices = batch["movieId"]
        ratings = batch["rating"]
        preds = self.forward(user_indices, item_indices)
        ratings = ratings.to(preds.dtype)
        mae = torch.nn.functional.l1_loss(preds, ratings)
        rmse = torch.sqrt(torch.nn.functional.mse_loss(preds, ratings))
        loss = rmse
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
        self.optimizer = Adam(self.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=1, threshold=0.001, verbose=True
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        }


class GeneralMatrixFactorizer(BaseCollaborativeFilter):
    def __init__(
        self,
        num_users,
        num_items,
        factor=16,
        learning_rate=0.01,
    ):
        super(GeneralMatrixFactorizer, self).__init__(
            num_users, num_items, factor, learning_rate
        )

        self.out = torch.nn.Linear(factor, 1)

    def get_last_latent(self, user_indices, item_indices):
        user_vecs = self.user_embedding(user_indices)
        item_vecs = self.item_embedding(item_indices)
        return user_vecs * item_vecs

    def forward(self, user_indices, item_indices):
        last_latent = self.get_last_latent(user_indices, item_indices)
        return self.out(last_latent)


class MultiLayerPerceptron(BaseCollaborativeFilter):
    def __init__(
        self,
        num_users,
        num_items,
        factor=16,
        num_layers=3,
        learning_rate=0.01,
    ):
        embedding_dim = factor * (2 ** (num_layers - 1))
        super(MultiLayerPerceptron, self).__init__(
            num_users, num_items, embedding_dim, learning_rate
        )

        self.mlp_layers = []
        for i in range(num_layers):
            input_size = factor * (2 ** (num_layers - i))
            # self.layers.append(nn.Dropout(p=self.dropout))
            self.mlp_layers.append(nn.Linear(input_size, input_size // 2))
            self.mlp_layers.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*self.mlp_layers)
        self.out = nn.Linear(factor, 1)

    def get_last_latent(self, user_indices, item_indices):
        user_vecs = self.user_embedding(user_indices)
        item_vecs = self.item_embedding(item_indices)
        return self.mlp_layers(torch.cat([user_vecs, item_vecs], dim=1))

    def forward(self, user_indices, item_indices):
        last_latent = self.get_last_latent(user_indices, item_indices)
        return self.out(last_latent).squeeze()


class NeuralMatrixFactorizer(BaseCollaborativeFilter):
    def __init__(
        self,
        num_users,
        num_items,
        factor=16,
        mlp_num_layers=3,
        learning_rate=0.01,
    ):
        super(NeuralMatrixFactorizer, self).__init__(
            num_users, num_items, factor, learning_rate
        )

        self.gmf_model = GeneralMatrixFactorizer(
            num_users, num_items, factor, learning_rate
        )
        self.mlp_model = MultiLayerPerceptron(
            num_users, num_items, factor, mlp_num_layers, learning_rate
        )
        self.out = torch.nn.Linear(2 * factor, 1)

    def forward(self, user_indices, item_indices):
        latent_gmf = self.gmf_model.get_last_latent(user_indices, item_indices)
        latent_mlp = self.mlp_model.get_last_latent(user_indices, item_indices)
        last_latent = torch.cat([latent_gmf, latent_mlp], dim=1)
        return self.out(last_latent).squeeze()
