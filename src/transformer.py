import torch
import torch.nn.functional as F


class FeedForward(torch.nn.Module):
    """
    FeedForward class.
    """

    def __init__(self, pitch_dim: int, hidden_dim: int, dropout: float = 0.2):
        """
        Constructor of the FeedForward class.

        Args:
            pitch_dim (int): Number of pitch dimensions.
            hidden_dim (int): Hidden dimension.
        """
        super().__init__()
        self.fc1 = torch.nn.Linear(pitch_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, pitch_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of the feedforward layer.

        Args:
            x (torch.Tensor): Input tensor [batch, ctx_size, pitch_dim].

        Returns:
            torch.Tensor: Output tensor [batch, ctx_size, pitch_dim].
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MaskedMultiHeadAttention(torch.nn.Module):
    """
    This class implements the masked multi-head attention layer.
    """

    def __init__(self, pitch_dim: int, num_heads: int, ctx_size: int):
        """
        Constructor of the MaskedMultiHeadAttention class.

        Args:
            pitch_dim (int): Number of pitch dimensions.
            num_heads (int): Number of heads.
            ctx_size (int): Context size.
        """
        super().__init__()
        self.pitch_dim = pitch_dim
        self.num_heads = num_heads
        self.head_dim = pitch_dim // num_heads

        self.W_q = torch.nn.Linear(pitch_dim, pitch_dim)
        self.W_k = torch.nn.Linear(pitch_dim, pitch_dim)
        self.W_v = torch.nn.Linear(pitch_dim, pitch_dim)

        self.lookahead_mask: torch.Tensor = torch.triu(
            torch.ones(ctx_size, ctx_size) * float("-inf"), diagonal=1
        )

    def forward(self, x: torch.Tensor, seq_length: int) -> torch.Tensor:
        """
        Compute the forward pass of the multi-head attention layer.

        Args:
            x (torch.Tensor): Input tensor [batch, ctx_size, pitch_dim].
            seq_length (int): Sequence length.

        Returns:
            torch.Tensor: Output tensor. [batch, ctx_size, pitch_dim].
        """
        # Q, K, V: [batch, sequence, pitch_dim]
        q: torch.Tensor = self.W_q(x)
        k: torch.Tensor = self.W_k(x)
        v: torch.Tensor = self.W_v(x)

        # split heads -> [batch, num_heads, sequence, query_dim]
        q = q.view(x.size(0), -1, self.num_heads, self.pitch_dim // self.num_heads)
        q = q.transpose(1, 2)
        k = k.view(x.size(0), -1, self.num_heads, self.pitch_dim // self.num_heads)
        k = k.transpose(1, 2)
        v = v.view(x.size(0), -1, self.num_heads, self.pitch_dim // self.num_heads)
        v = v.transpose(1, 2)

        # attention: [batch, num_heads, sequence, sequence]
        attention: torch.Tensor = torch.matmul(q, k.transpose(-2, -1)) / (
            self.pitch_dim**0.5
        )

        # Apply mask. If training, apply lookahead mask, else apply padding mask
        if self.training:
            attention = attention + self.lookahead_mask
        else:
            attention[:, :, :, :seq_length] = float("-inf")

        # softmax: [batch, num_heads, sequence, sequence]
        attention = F.softmax(attention, dim=-1)

        # output: [batch, num_heads, sequence, query_dim]
        output = torch.matmul(attention, v)

        # merge heads -> [batch, sequence, pitch_dim]
        output = output.transpose(1, 2).contiguous().view(x.size(0), -1, self.pitch_dim)
        return output


class PositionalEncoding(torch.nn.Module):
    """
    This class implements the positional encoding layer.
    """

    def __init__(self, pitch_dim: int, max_len: int):
        """
        Constructor of the PositionalEncoding class (taken from pytorch documentation).

        Args:
            pitch_dim (int): Number of pitch dimensions.
            max_len (int): Maximum length of the sequence.
        """
        super().__init__()
        self.pitch_dim = pitch_dim
        self.max_len = max_len

        # Compute the positional encoding
        pe = torch.zeros(max_len, pitch_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, pitch_dim, 2).float()
            * -(torch.log(torch.tensor(10000.0)))
            / pitch_dim
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1).unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of the positional encoding layer.

        Args:
            x (torch.Tensor): Input tensor [batch, ctx_size, pitch_dim].

        Returns:
            torch.Tensor: Output tensor. [batch, ctx_size, pitch_dim].
        """
        return x + self.pe[:, :, : x.size(2)]


class Decoder(torch.autograd.Function):
    """
    Class for the Transformer model.
    """

    def __init__(
        self,
        pitch_dim: int,
        num_heads: int,
        hidden_dim: int,
        ctx_size: int = 160,
    ):
        """
        Constructor of the Decoder class.

        Args:
            pitch_dim (int): Number of pitch dimensions.
            num_heads (int): Number of heads.
            hidden_dim (int): Hidden dimension.
            ctx_size (int): Context size.
        """
        super().__init__()

        self.pitch_dim = pitch_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.ctx_size = ctx_size

        self.masked_multi_head_attention: MaskedMultiHeadAttention = (
            MaskedMultiHeadAttention(pitch_dim, num_heads, ctx_size=self.ctx_size),
        )
        self.layer_norm: torch.nn.LayerNorm = torch.nn.LayerNorm(pitch_dim)
        self.fc: FeedForward = FeedForward(pitch_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of the decoder.

        Args:
            x (torch.Tensor): Input tensor [batch, ctx_size, pitch_dim].

        Returns:
            torch.Tensor: Output tensor [batch, ctx_size, pitch_dim].
        """

        # Apply LayerNorm
        output = self.layer_norm(output)

        # Apply the multi-head attention layers
        output = self.masked_multi_head_attention(output, self.ctx_size)

        # Add the inputs
        output = output + x

        # Save the inputs for later
        inputs = output

        # Apply LayerNorm
        output = self.layer_norm(output)

        # Apply the feed-forward layer
        output = self.fc(output)

        # Add the inputs
        output = output + inputs

        return output


class MyModel(torch.nn.Module):
    def __init__(
        self,
        pitch_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        ctx_size: int = 160,
    ):
        super().__init__()
        self.positional_encoding: PositionalEncoding = PositionalEncoding(
            pitch_dim=pitch_dim, max_len=100
        )
        self.layers = torch.nn.Sequential(
            [
                Decoder(
                    pitch_dim=pitch_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    ctx_size=ctx_size,
                )
                for _ in range(num_layers)
            ]
        )

        self.linear_out = torch.nn.Linear(pitch_dim, 1)
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        pass
