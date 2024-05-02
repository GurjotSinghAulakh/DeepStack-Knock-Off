import lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DeepstackNN(pl.LightningModule):
    def __init__(self, range_size: int, public_info_size: int):
        super().__init__()
        self.range_size = range_size
        self.public_info_size = public_info_size

        # Input is:
        # range_size * 2 (range vectors for player 1 and player 2)
        # public_info_size (public cards information)
        # 1 (pot size)
        self.fc1 = nn.Linear(range_size * 2 + public_info_size + 1, 1400)
        self.fc2 = nn.Linear(1400, 1000)
        self.fc3 = nn.Linear(1000, 700)
        self.value_output = nn.Linear(700, range_size * 2)
        # Output is:
        # range_size * 2 (value vectors for player 1 and player 2)

    def predict_values(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Utility wrapper function to predict values from input tensor

        args:
        x: torch.Tensor - input tensor

        returns:
        tuple[np.ndarray, np.ndarray] - predicted values for player 1 and player 2
        """
        v1, v2, _ = self(x)  # Discarding ev_sum here, that is only used for traning.
        return v1.detach().numpy().flatten(), v2.detach().numpy().flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feed forward function

        args:
        x: torch.Tensor - input tensor

        returns:
        torch.Tensor - output tensor
        """
        inputcopy = x.clone()
        features = self.fc1(x)
        features = torch.relu(features)
        features = self.fc2(features)
        features = torch.relu(features)
        features = self.fc3(features)
        features = torch.relu(features)

        values = self.value_output(features)
        v1, v2 = values.split([self.range_size, self.range_size], dim=1)

        r1, r2, _ = inputcopy.split([self.range_size, self.range_size, self.public_info_size + 1], dim=1)

        p1_ev = torch.sum(r1 * v1, dim=1, keepdim=True)
        p2_ev = torch.sum(r2 * v2, dim=1, keepdim=True)

        ev_sum = p1_ev + p2_ev

        return v1, v2, ev_sum

    def configure_optimizers(self):
        """
        Configure the optimizer and (deprecated) scheduler
        """
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.0001)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        # return [optimizer], [scheduler]
        return optimizer

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Training step for the model, calculates loss and logs it

        args:
        batch: torch.Tensor - input tensor

        returns:
        torch.Tensor - loss
        """
        x, p1_target, p2_target = batch.split([                        # The Tensor is structured as follows:
            self.range_size * 2 + self.public_info_size + 1,           # range_size          |   (276-len float range vector for player 1)
            self.range_size,                                           # + range_size        |   (276-len float range vector for player 1)
            self.range_size,                                           # + public_info_size  |   (5-3 ints, public cards)
        ], dim=1)                                                      # + 1                 |   (1 int, pot size)
        v1, v2, ev_sum = self(x)                                       # + range_size        |   (276-len float value vectors for player 1)
        loss = self.custom_loss(v1, v2, ev_sum, p1_target, p2_target)  # + range_size        |   (276-len float value vectors for player 2)
        self.log('train_loss', loss)                                   # = 1108 - 1111
        return loss

    def validation_step(self, batch: torch.Tensor) -> None:
        """
        Validation step for the model, calculates loss and logs it

        args:
        batch: torch.Tensor - input tensor

        returns:
        None
        """
        x, p1_target, p2_target = batch.split([
            self.range_size * 2 + self.public_info_size + 1,
            self.range_size,
            self.range_size,
        ], dim=1)
        v1, v2, ev_sum = self(x)

        loss = self.custom_loss(v1, v2, ev_sum, p1_target, p2_target)
        self.log('val_loss', loss)

    def custom_loss(self, v1: torch.Tensor, v2: torch.Tensor, ev_sum: torch.Tensor, p1_target: torch.Tensor, p2_target: torch.Tensor) -> torch.Tensor:
        """
        Define a custom loss function. This is the sum of the MSE loss for the value outputs and the expected value sum, which should be zero.

        args:
        v1: torch.Tensor - output tensor for player 1
        v2: torch.Tensor - output tensor for player 2
        ev_sum: torch.Tensor - expected value sum
        p1_target: torch.Tensor - target tensor for player 1
        p2_target: torch.Tensor - target tensor for player 2

        returns:
        torch.Tensor - total loss
        """
        loss_p1 = nn.functional.mse_loss(v1, p1_target)
        loss_p2 = nn.functional.mse_loss(v2, p2_target)
        loss_ev = nn.functional.mse_loss(ev_sum, torch.zeros_like(ev_sum))

        total_loss = loss_p1 + loss_p2 + loss_ev  # + mask_pen_v1 + mask_pen_v2
        return total_loss
