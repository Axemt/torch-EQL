import torch
from torch import nn
import numpy as np
from . import EQL
from sys import modules
from inspect import isclass, getmembers


class L0GenericLoss(nn.Module):

    def __init__(
            self,
            base_loss: nn.Module, 
            L0_lambda: float=1e-2,
        ) -> None:
        super().__init__()

        self.base_loss = base_loss()
        self.L0_lambda = L0_lambda

    def forward(self, model: EQL, y: torch.Tensor, yhat_1: torch.Tensor):

        loss = self.base_loss(y, yhat_1)

        L0_reg = 0
        if self.L0_lambda != 0:
            reg_loss = model.l0_loss()
            L0_reg = self.L0_lambda * reg_loss
        
        return loss + L0_reg

class L0MSELoss(L0GenericLoss):

    def __init__(self, **kwargs) -> None:
        super().__init__(nn.MSELoss, **kwargs)

class L0DifferentialEntropyLoss(L0GenericLoss):

    def __init__(self, **kwargs):
        super().__init__(DifferentialEntropyLoss, **kwargs)


class L0ExpRMSE(L0GenericLoss):

    def __init__(self, **kwargs):
        super().__init__(ExpRMSE, **kwargs)

def vasicek_differential_entropy(x: torch.Tensor, eps: float = 1e-10) -> float:

    n = x.shape[0]
    window_size = int(np.floor( np.sqrt(n) + 0.5 ))

    # Ensure X is a 1D tensor - it is smaller and faster in mem
    X = x[:, -1]
    
    # Compute the differences X(i+m) - X(i-m)
    indices = torch.arange(window_size, n - window_size, dtype=torch.int).to('cpu')
    X_diff = X[indices + window_size] - X[indices - window_size]
    
    return torch.mean(
        torch.log((n / (2 * window_size)) * X_diff.abs() + eps)
    )

class DifferentialEntropyLoss(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, y: torch.Tensor, y_hat1: torch.Tensor):

        return vasicek_differential_entropy(y - y_hat1)
    
class ExpRMSE(nn.Module):

    def __init__(self, *args, alpha: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> float:

        return torch.exp( self.alpha + (y - y_hat) ** 2 )


LOSSES = {
    name.lower() : obj
    for name, obj in getmembers(
        modules[__name__], 
        lambda obj: isclass(obj) and obj.__module__ == __name__)
}