import torch
import torch.nn.functional as F


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

        self.output = torch.nn.Linear(pitch_dim, pitch_dim)

    def forward(self, x: torch.Tensor, seq_length: int) -> torch.Tensor:
        """
        Compute the forward pass of the multi-head attention layer.

        Args:
            x (torch.Tensor): Input tensor [batch, ctx_size, pitch_dim].

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


class Decoder(torch.autograd.Function):
    """
    Class for the Transformer model.
    """

    def __init__(
        self, input_dim, output_dim, num_heads, num_layers, hidden_dim, dropout=0.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.positional_encoding = self.get_positional_encoding()

    def get_positional_encoding(self):
        """
        Get positional encoding for the transformer model.

        Returns:
            torch.Tensor: Positional encoding.
        """


class MyModel(torch.nn.Module):
    def __init__(
        self, input_dim, output_dim, num_heads, num_layers, hidden_dim, dropout=0.0
    ):
        super().__init__()
        self.inputs = torch.nn.Linear(input_dim, hidden_dim)
        self.outputs = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        pass
