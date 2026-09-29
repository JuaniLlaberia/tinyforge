import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 std: float = 0.02,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        """
        Initializes the Embedding Module.

        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors, i.e., 𝑑model
            std (float): Standard deviation for the truncated normal init
            device (torch.device | None) Device to store the parameters on
            dtype (torch.dtype | None) Data type of the parameters
        """
        super().__init__()
        self.embedding_matrix = nn.Parameter(torch.empty(num_embeddings,
                                                         embedding_dim,
                                                         device=device,
                                                         dtype=dtype))

        with torch.no_grad():
            nn.init.trunc_normal_(self.embedding_matrix, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        """
        return self.embedding_matrix[token_ids]