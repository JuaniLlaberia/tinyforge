import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 std: float = 0.02,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        """
        Initializes Linear Module.

        Args:
            in_features (int): Final dimension of the input
            out_features (int): Final dimension of the output
            std (float): Standard deviation for the truncated normal init
            device (torch.device | None) Device to store the parameters on
            dtype (torch.dtype | None) Data type of the parameters
        """
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_features,
                                          in_features,
                                          device=device,
                                          dtype=dtype))

        with torch.no_grad():
            nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass by applying a linear transformation to input x.
        """
        return x @ self.W.T