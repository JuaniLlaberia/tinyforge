import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self,
                d_model: int,
                eps: float = 1e-5,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None):
        """
        Initializes the RMSNorm module.

        Args:
            d_model (int): Hidden dimension of the model
            eps (float = 1e-5): Epsilon value for numerical stability
            device (torch.device | None): Device to store the parameters on
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS normalization followed by the learned gain.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).
        Returns:
            torch.Tensor: Normalized tensor of the same shape as x.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.g

        return result.to(in_dtype)