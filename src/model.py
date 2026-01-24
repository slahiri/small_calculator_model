"""
Calculator LLM Model Architecture

A tiny decoder-only transformer for solving English math problems.
Built from scratch following: https://sid.sh/learn/build-your-first-llm

Model: 105K parameters, 64 embed_dim, 4 heads, 2 layers

Architecture Overview:
1. TokenEmbedding: Convert token IDs to dense vectors + positional info
2. TransformerBlocks: Process sequences using attention and feed-forward layers
3. Output Projection: Convert back to vocabulary probabilities
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Transformers have no built-in notion of position - they see all tokens
    at once. Positional encoding adds position information to embeddings
    so the model knows the order of tokens.

    We use sinusoidal encoding (from "Attention Is All You Need"):
    - PE(pos, 2i) = sin(pos / 10000^(2i/d))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    This creates unique patterns for each position that the model can
    learn to interpret.
    """

    def __init__(self, embed_dim: int, max_seq_len: int = 512, dropout: float = 0.1):
        """
        Args:
            embed_dim: Dimension of embeddings (must match token embeddings)
            max_seq_len: Maximum sequence length to support
            dropout: Dropout probability for regularization
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix [max_seq_len, embed_dim]
        pe = torch.zeros(max_seq_len, embed_dim)

        # Position indices: [0, 1, 2, ..., max_seq_len-1]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Compute the division term for the sinusoidal formula
        # This creates different frequencies for different dimensions
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions

        # Add batch dimension: [1, max_seq_len, embed_dim]
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Token embeddings [batch_size, seq_len, embed_dim]

        Returns:
            Embeddings with position info added [batch_size, seq_len, embed_dim]
        """
        # Add positional encoding (broadcasting handles batch dimension)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """
    Token embedding with positional encoding.

    This is the input layer of our transformer. It:
    1. Converts token IDs to dense vectors (learnable embeddings)
    2. Scales embeddings by sqrt(embed_dim) for stable training
    3. Adds positional encoding so the model knows token order

    Example:
        Input: [1, 6, 32, 7]  (token IDs for "[START] two plus three")
        Output: [batch, 4, 64]  (4 tokens, each as a 64-dim vector)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_size: Number of tokens in vocabulary (36 for our calculator)
            embed_dim: Dimension of embedding vectors (64 in our model)
            max_seq_len: Maximum sequence length to support
            dropout: Dropout probability for regularization
        """
        super().__init__()
        # Learnable embedding matrix: [vocab_size, embed_dim]
        # Each row is the embedding vector for one token
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding to add position information
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)

        # Scaling factor for embeddings (from "Attention Is All You Need")
        # This helps with gradient flow during training
        self.scale = math.sqrt(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings with positional information.

        Args:
            x: Token IDs [batch_size, seq_len]

        Returns:
            Embeddings [batch_size, seq_len, embed_dim]
        """
        # Look up embeddings and scale
        x = self.token_embedding(x) * self.scale
        # Add positional encoding
        return self.pos_encoding(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Attention allows each token to "look at" all other tokens and decide
    which ones are relevant. Multi-head attention runs multiple attention
    operations in parallel, each focusing on different aspects.

    The attention formula is:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Where:
    - Q (Query): What am I looking for?
    - K (Key): What do I contain?
    - V (Value): What information do I have?

    For "two plus three equals":
    - "equals" might attend strongly to "two", "plus", and "three"
    - This helps it predict the next token ("five")
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            embed_dim: Dimension of embeddings (must be divisible by num_heads)
            num_heads: Number of attention heads (4 in our model)
            dropout: Dropout probability for attention weights
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Each head sees embed_dim/num_heads dims

        # Linear projections to create Q, K, V from input
        self.q_proj = nn.Linear(embed_dim, embed_dim)  # Query projection
        self.k_proj = nn.Linear(embed_dim, embed_dim)  # Key projection
        self.v_proj = nn.Linear(embed_dim, embed_dim)  # Value projection

        # Output projection to combine heads
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)  # Scaling factor for dot product

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head self-attention.

        Args:
            x: Input embeddings [batch_size, seq_len, embed_dim]
            mask: Optional causal mask to prevent attending to future tokens

        Returns:
            (output, attention_weights) tuple
            - output: [batch_size, seq_len, embed_dim]
            - attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: Project input to Q, K, V and reshape for multi-head
        # Shape: [batch, seq, embed] -> [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        Q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Step 2: Compute attention scores
        # scores[i,j] = how much token i should attend to token j
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Step 3: Apply causal mask (if provided)
        # This prevents tokens from attending to future tokens
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Step 4: Apply softmax to get attention weights (sum to 1)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Step 5: Apply attention weights to values
        attn_output = torch.matmul(attn_weights, V)

        # Step 6: Reshape and project output
        # [batch, heads, seq, head_dim] -> [batch, seq, embed]
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )
        return self.out_proj(attn_output), attn_weights


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    After attention, each position goes through this two-layer network:
    1. Linear: embed_dim -> ff_dim (expand)
    2. ReLU activation (non-linearity)
    3. Linear: ff_dim -> embed_dim (compress back)

    This allows the model to transform the attention output and learn
    more complex patterns. The "position-wise" means each position
    is processed independently (but with shared weights).
    """

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        """
        Args:
            embed_dim: Input/output dimension (64 in our model)
            ff_dim: Hidden dimension (256 in our model, typically 4x embed_dim)
            dropout: Dropout probability
        """
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)   # Expand: 64 -> 256
        self.linear2 = nn.Linear(ff_dim, embed_dim)   # Compress: 256 -> 64
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward transformation.

        Args:
            x: Input [batch_size, seq_len, embed_dim]

        Returns:
            Output [batch_size, seq_len, embed_dim]
        """
        # Expand -> ReLU -> Dropout -> Compress
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    A single transformer decoder block.

    Each block contains:
    1. Multi-head self-attention (with residual connection + layer norm)
    2. Feed-forward network (with residual connection + layer norm)

    The residual connections (x + sublayer(x)) help with gradient flow
    during training. Layer normalization stabilizes training.

    Our model stacks 2 of these blocks.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: Embedding dimension (64)
            num_heads: Number of attention heads (4)
            ff_dim: Feed-forward hidden dimension (256)
            dropout: Dropout probability
        """
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)  # Normalize after attention
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)  # Normalize after feed-forward
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process input through attention and feed-forward layers.

        Args:
            x: Input [batch_size, seq_len, embed_dim]
            mask: Optional causal mask

        Returns:
            (output, attention_weights) tuple
        """
        # Sub-layer 1: Multi-head attention with residual connection
        attn_output, attn_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # Add & Norm

        # Sub-layer 2: Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))  # Add & Norm

        return x, attn_weights


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Create a causal mask to prevent attending to future tokens.

    In language modeling, when predicting token t, we can only see
    tokens 0, 1, ..., t-1. The causal mask enforces this by setting
    future positions to 0 (which becomes -inf after masking).

    For seq_len=4, the mask looks like:
        [[1, 0, 0, 0],   <- token 0 sees only itself
         [1, 1, 0, 0],   <- token 1 sees tokens 0, 1
         [1, 1, 1, 0],   <- token 2 sees tokens 0, 1, 2
         [1, 1, 1, 1]]   <- token 3 sees all tokens

    Args:
        seq_len: Length of the sequence

    Returns:
        Causal mask [1, 1, seq_len, seq_len]
    """
    # Create lower triangular matrix (1s on and below diagonal)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    # Add batch and head dimensions for broadcasting
    return mask.unsqueeze(0).unsqueeze(0)


class CalculatorLLM(nn.Module):
    """
    A tiny transformer LLM for solving English math problems.

    Architecture:
    1. TokenEmbedding: token IDs -> embeddings with position
    2. TransformerBlocks (x2): attention + feed-forward
    3. LayerNorm: normalize final output
    4. Output projection: embeddings -> vocabulary logits

    Model size: ~105K parameters
    - Vocab: 36 tokens
    - Embedding: 64 dimensions
    - Heads: 4
    - Layers: 2
    - FF dimension: 256

    Training:
    - Input: "two plus three equals" (token IDs)
    - Target: "plus three equals five" (shifted by 1)
    - The model learns to predict the next token at each position
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_size: Number of tokens in vocabulary (36)
            embed_dim: Embedding dimension (64)
            num_heads: Number of attention heads (4)
            num_layers: Number of transformer blocks (2)
            ff_dim: Feed-forward hidden dimension (256)
            max_seq_len: Maximum sequence length (16)
            dropout: Dropout probability
        """
        super().__init__()
        self.max_seq_len = max_seq_len

        # Input: Convert tokens to embeddings
        self.embedding = TokenEmbedding(vocab_size, embed_dim, max_seq_len, dropout)

        # Middle: Stack of transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        # Output: Final layer norm and projection to vocabulary
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Token IDs [batch_size, seq_len]
            mask: Optional causal mask (auto-created if None)

        Returns:
            Logits [batch_size, seq_len, vocab_size]
            Each position outputs a probability distribution over the vocabulary
        """
        # Create causal mask if not provided
        if mask is None:
            seq_len = x.size(1)
            mask = create_causal_mask(seq_len).to(x.device)

        # Token IDs -> Embeddings with positional encoding
        x = self.embedding(x)

        # Pass through transformer blocks
        for layer in self.layers:
            x, _ = layer(x, mask)

        # Final normalization and projection to vocabulary
        x = self.norm(x)
        return self.output_proj(x)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
