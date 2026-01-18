"""
Embedding layers for the Calculator LLM.

Contains:
- TokenEmbedding: Converts token IDs to embedding vectors
- PositionalEncoding: Adds position information to embeddings
- InputEmbedding: Complete pipeline combining token and positional embeddings
"""

import math
import torch
import torch.nn as nn


# Model configuration for calculator LLM
VOCAB_SIZE = 36
EMBED_DIM = 64
MAX_SEQ_LEN = 32


class TokenEmbedding(nn.Module):
    """
    Converts token IDs to embedding vectors.

    This is a simple lookup table that maps each token ID to a learned
    vector of size embed_dim.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Args:
            vocab_size: Number of tokens in vocabulary (36 for our calculator)
            embed_dim: Dimension of embedding vectors (64 for our model)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # The embedding layer: a lookup table of shape [vocab_size, embed_dim]
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for token IDs.

        Args:
            token_ids: Tensor of shape [batch_size, seq_len]

        Returns:
            Tensor of shape [batch_size, seq_len, embed_dim]

        Example:
            >>> emb = TokenEmbedding(36, 64)
            >>> ids = torch.tensor([[5, 31, 6]])
            >>> emb(ids).shape
            torch.Size([1, 3, 64])
        """
        # embedding layer handles the lookup automatically
        return self.embedding(token_ids)


class PositionalEncoding(nn.Module):
    """
    Adds position information to embeddings.

    Uses learned positional embeddings, which are simpler than sinusoidal
    and work well for fixed-length sequences.
    """

    def __init__(self, max_seq_len: int, embed_dim: int):
        """
        Args:
            max_seq_len: Maximum sequence length (32 for our model)
            embed_dim: Dimension of embeddings (must match TokenEmbedding)
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Learned positional embeddings: each position gets its own vector
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to embeddings.

        Args:
            embeddings: Tensor of shape [batch_size, seq_len, embed_dim]

        Returns:
            Tensor of shape [batch_size, seq_len, embed_dim] with position info added

        Example:
            >>> pos = PositionalEncoding(32, 64)
            >>> emb = torch.randn(2, 5, 64)
            >>> pos(emb).shape
            torch.Size([2, 5, 64])
        """
        batch_size, seq_len, _ = embeddings.shape

        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=embeddings.device)

        # Look up positional embeddings: [seq_len, embed_dim]
        pos_embeddings = self.position_embedding(positions)

        # Add positional embeddings to token embeddings
        # Broadcasting handles the batch dimension automatically
        return embeddings + pos_embeddings


class InputEmbedding(nn.Module):
    """
    Complete input pipeline: token IDs -> embeddings with position.

    Combines TokenEmbedding and PositionalEncoding into a single module.
    This is what the transformer encoder will use as input.
    """

    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int):
        """
        Args:
            vocab_size: Number of tokens in vocabulary
            embed_dim: Dimension of embedding vectors
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(max_seq_len, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to position-aware embeddings.

        Args:
            token_ids: Tensor of shape [batch_size, seq_len]

        Returns:
            Tensor of shape [batch_size, seq_len, embed_dim]

        Example:
            >>> inp = InputEmbedding(36, 64, 32)
            >>> ids = torch.tensor([[5, 31, 6]])
            >>> inp(ids).shape
            torch.Size([1, 3, 64])
        """
        # Step 1: Get token embeddings
        token_emb = self.token_embedding(token_ids)

        # Step 2: Add positional encoding
        output = self.positional_encoding(token_emb)

        return output
