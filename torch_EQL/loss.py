import torch
from torch import nn

class L1GenericLoss(nn.Module):

    def __init__(self, base_loss: nn.Module, L1_lambda: float=1e-7) -> None:
        super().__init__()

        self.base_loss = base_loss()
        self.L1_lambda = L1_lambda

    def __call__(self, model_params, y, yhat_1):

        loss = self.base_loss(y, yhat_1)

        weights = torch.cat([x.view(-1) for x in model_params])
        L1_reg = self.L1_lambda * torch.norm(weights, 1)

        return loss + L1_reg


class L1MSELoss(L1GenericLoss):

    def __init__(self, **kwargs) -> None:
        super().__init__(nn.MSELoss, **kwargs)
