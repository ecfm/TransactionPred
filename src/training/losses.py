import torch
import torch.nn as nn
from src.utils.registry import create_registry

LossRegistry = create_registry()

@LossRegistry.register("no_op")
class NoOpLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self,predictions, targets):
        return torch.tensor(0.0)
    
@LossRegistry.register("l1")
class L1Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, predictions, targets):
        return self.l1_loss(predictions, targets)

@LossRegistry.register("l2")
class L2Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l2_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        return self.l2_loss(predictions, targets)