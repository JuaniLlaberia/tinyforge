import torch
from torch import nn

class RoPE(nn.Module):
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device: torch.device | None = None):
        """
        Initializes the RoPE module and create buffers if needed.

        Args:
            theta (float): Θ value for the RoPE
            d_k (int): dimension of query and key vectors
            max_seq_len (int):  Maximum sequence length that will be input
            device (torch.device | None = None): Device to store the buffer on
        """
        super().__init__()

        theta_angle = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        seq_idx = torch.arange(max_seq_len, device=device)
        idx_theta = torch.outer(seq_idx, theta_angle)

        self.register_buffer("cosine", idx_theta.cos(), persistent=False)
        self.register_buffer("sine", idx_theta.sin(), persistent=False)

    def forward(self,
                x: torch.Tensor,
                token_positions: torch.Tensor) -> torch.Tensor:
        """
        Applies rotary position embeddings to the even/odd feature pairs of x.

        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_k).
            token_positions (torch.Tensor): Position indices of shape (..., seq_len).
        Returns:
            torch.Tensor: Tensor of the same shape as x with rotary embeddings applied.
        """
        x_even, x_odd = x[..., 0::2], x[...,1::2]

        cos = self.cosine[token_positions]
        sin = self.sine[token_positions]

        extra_dims = x.ndim - cos.ndim

        for _ in range(extra_dims):
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        x_even_b = x_even * cos - x_odd * sin
        x_odd_b  = x_even * sin  + x_odd  * cos

        stacked = torch.stack([x_even_b, x_odd_b], dim=-1)
        interleaved = stacked.flatten(-2)

        return interleaved