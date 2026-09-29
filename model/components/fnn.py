import torch
from torch import nn

from .linear import Linear

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Initializes Position-wise Feed-Forward Network (FFN) using the SwiGLU activation.

        Args:
            d_model (int): Dimensionality of the input and output token embeddings.
            d_ff (int): Hidden dimension of the feed-forward network.
        """
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the SiLU (Sigmoid Linear Unit) activation function.

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Tensor with the SiLU activation applied element-wise.
        """
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the position-wise SwiGLU feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model), typically
                (batch_size, seq_len, d_model).
        Returns:
            torch.Tensor: Tensor of the same shape as the input with the transformed
            token representations.
        """
        return self.w2(self._silu(self.w1(x)) * self.w3(x))