import torch
from torch import Tensor
from torch.nn import Module

class Reshape(Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x: Tensor) -> Tensor:
        return x.view(self.shape)

class Squeeze(Module):
    def __init__(self, dim: int=None):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.squeeze(self.dim)

class UnitNorm(Module):
    def __init__(self, dim: int=None):
        super(UnitNorm, self).__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        return (x - torch.mean(x, dim=self.dim)) / torch.std(x, dim=self.dim)

class RBF(Module):
    """
    Radial Basis Function
    Src: https://github.com/PaddlePaddle/PaddleHelix/blob/dev/pahelix/networks/basic_block.py#L71
    """
    def __init__(self, centers: list, gamma: float, dtype=torch.float32):
        super(RBF, self).__init__()
        self.centers = torch.reshape(torch.tensor(centers, dtype=dtype), [1, -1]).cuda()
        self.gamma = gamma
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x(tensor): (-1, 1).
        Returns:
            y(tensor): (-1, n_centers)
        """
        x = torch.reshape(x, [-1, 1]).cuda()
        return torch.exp(-self.gamma * torch.square(x - self.centers))
