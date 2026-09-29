import torch
from torch import nn

from .rsm_norm import RMSNorm
from .attention import CausalMultiheadSelfAttention
from .fnn import PositionwiseFeedForward

class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int,
                 theta: float,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        """
        Initializes a pre-norm Transformer block (attention + feed-forward with residuals).

        Args:
            d_model (int): Model hidden dimension.
            num_heads (int): Number of attention heads.
            d_ff (int): Hidden dimension of the feed-forward network.
            max_seq_len (int): Maximum sequence length supported by RoPE.
            theta (float): Θ value for RoPE.
            device (torch.device | None): Device to store the parameters on.
            dtype (torch.dtype | None): Data type of the parameters.
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.rmsnorm_1 = RMSNorm(d_model=self.d_model,
                               device=device,
                               dtype=dtype)
        self.attn = CausalMultiheadSelfAttention(d_model=self.d_model,
                                                 num_heads=self.num_heads,
                                                 theta=theta,
                                                 max_seq_len=max_seq_len,
                                                 device=device,
                                                 dtype=dtype)

        self.rmsnorm_2 = RMSNorm(d_model=self.d_model,
                               device=device,
                               dtype=dtype)
        self.ffn = PositionwiseFeedForward(d_model=self.d_model,
                                           d_ff=self.d_ff,
                                           device=device,
                                           dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Applies one Transformer block: attention sub-layer then feed-forward sub-layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).
            token_positions (torch.Tensor): Position indices used for RoPE.
        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, d_model).
        """
        h = self.rmsnorm_1(x)
        x = x + self.attn(h, token_positions)

        h = self.rmsnorm_2(x)
        x = x + self.ffn(h)

        return x