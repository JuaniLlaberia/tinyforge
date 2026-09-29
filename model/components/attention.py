import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

from .linear import Linear
from .rope import RoPE

class CausalMultiheadSelfAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 theta: float,
                 max_seq_len: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        """
        Initializes causal multi-head self-attention with rotary position embeddings.

        Args:
            d_model (int): Model hidden dimension.
            num_heads (int): Number of attention heads.
            theta (float): Θ value for RoPE.
            max_seq_len (int): Maximum sequence length supported by RoPE.
            device (torch.device | None): Device to store the parameters on.
            dtype (torch.dtype | None): Data type of the parameters.
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device

        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_k = d_model // num_heads

        # Query projection
        self.q_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        # Key projection
        self.k_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        # Value projection
        self.v_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        # Output projection
        self.o_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)

        # Setup RoPE
        self.rope = RoPE(theta=theta,
                         d_k=self.d_k,
                         max_seq_len=max_seq_len,
                         device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Applies causal multi-head self-attention with RoPE to the input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).
            token_positions (torch.Tensor): Position indices used for RoPE.
        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, d_model).
        """
        # Get the sequence length
        seq_len = x.size(-2)

        q = self.q_proj(x) # (batch, seq_len, d_model)
        k = self.k_proj(x) # (batch, seq_len, d_model)
        v = self.v_proj(x) # (batch, seq_len, d_model)

        # Reshape into: (batch, num_heads, seq_len, d_k)
        shape = q.shape[:-1] + (self.num_heads, self.d_k)
        q = q.view(shape).transpose(-3, -2)
        k = k.view(shape).transpose(-3, -2)
        v = v.view(shape).transpose(-3, -2)

        # Apply RoPE
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        # Create Mask
        mask = torch.tril(torch.ones((seq_len, seq_len),
                                     device=x.device,
                                     dtype=torch.bool))

        attention = scaled_dot_product_attention(q, k, v, mask).transpose(-3, -2).contiguous().view(x.shape[:-1] + (self.d_model,))

        return self.o_proj(attention)