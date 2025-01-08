import torch
import torch.nn as nn


class WeightedLoss(nn.Module):
    def __init__(self, loss, mode):
        super(WeightedLoss, self).__init__()
        assert loss in ['mse', 'mae']
        assert mode in ['log', 'linear']
        self.loss = loss
        self.mode = mode

        if self.loss == 'mse':
            self.a = 10.6009
            self.b = 0.2259
            self.c = -0.7640
        elif self.loss == 'mae':
            self.a = 3.2408
            self.b = 0.1134
            self.c = -0.0694

    def forward(self, yhat, y, snr):
        """
        Compute the weighted MSE loss.

        Args:
            yhat (torch.Tensor): Predicted values (batch_size, ...)
            y (torch.Tensor): Target values (batch_size, ...)
            weights (torch.Tensor): Weights for each sample in the batch (batch_size,)

        Returns:
            torch.Tensor: Weighted MSE loss.
        """
        # Ensure weights shape matches batch size
        weights = self.compute_weights(snr)
        if weights.dim() == 1:
            weights = weights.view(-1, *[1] * (yhat.dim() - 1))  # Expand dimensions if needed

        if self.loss == 'mse':
            error = (yhat - y) ** 2  # Element-wise squared error
        elif self.loss == 'mae':
            error = torch.abs(y - yhat)
        else:
            raise NotImplementedError
        weighted_error = weights * error  # Apply weights to the squared error
        return weighted_error.mean()  # Mean over all elements

    def compute_weights(self, x):
        weight = self.a * torch.exp(self.b * x) + self.c
        if self.mode == 'log':
            return torch.log10(weight) + 1
        else:
            return weight
