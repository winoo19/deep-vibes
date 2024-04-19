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

        # self.lookahead_mask: torch.Tensor = torch.triu(
        #     torch.ones(ctx_size, ctx_size) * float("-inf"), diagonal=1
        # ).to(torch.device("cuda"))
        self.lookahead_mask: torch.Tensor = torch.triu(
            torch.ones(ctx_size, ctx_size) * float("-inf"), diagonal=1
        ).to(torch.device("cpu"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of the multi-head attention layer.

        Args:
            x (torch.Tensor): Input tensor [batch, ctx_size, pitch_dim].
            seq_length (int): Sequence length.

        Returns:
            torch.Tensor: Output tensor. [batch, ctx_size, pitch_dim].
        """
        # Split inputs
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

        attention.to(x.device)

        # Apply mask. If training, apply lookahead mask, else apply padding mask
        attention = attention + self.lookahead_mask

        # softmax: [batch, num_heads, sequence, sequence]
        attention = F.softmax(attention, dim=-1)

        # output: [batch, num_heads, sequence, query_dim]
        output = torch.matmul(attention, v)

        # merge heads -> [batch, sequence, pitch_dim]
        output = output.transpose(1, 2).contiguous().view(x.size(0), -1, self.pitch_dim)
        return output

    def inference(self, x: torch.Tensor, seq_length: int) -> torch.Tensor:
        """
        Method that infers a new sequence from the input tensor.

        Args:
            x (torch.Tensor): Input tensor [batch, ctx_size, pitch_dim].
            seq_length (int): Sequence length.

        Returns:
            torch.Tensor: Output tensor [batch, ctx_size, pitch_dim].
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

        attention.to(x.device)

        # Apply mask. If training, apply lookahead mask, else apply padding mask
        attention[:, :, :, seq_length:] = float("-inf")

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
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of the positional encoding layer.

        Args:
            x (torch.Tensor): Input tensor [batch, ctx_size, pitch_dim].

        Returns:
            torch.Tensor: Output tensor. [batch, ctx_size, pitch_dim].
        """
        return x + self.pe


class CustomBCELoss(torch.nn.Module):
    """
    Custom binary cross-entropy loss function.
    """

    def __init__(self, loss: torch.nn.Module) -> None:
        super(CustomBCELoss, self).__init__()
        self.loss = loss

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of the BCE loss function.

        Args:
            x (torch.Tensor): Output tensor. [batch, ctx_size, pitch_dim].
            y (torch.Tensor): Target tensor. [batch, ctx_size, pitch_dim].

        Returns:
            torch.Tensor: Loss value.
        """
        # Compare element i of x with element i+1 of y
        assert x.shape == y.shape
        return self.loss(x[:, :-1], y[:, 1:])


class Sigmoid(torch.nn.Module):
    """
    Custom sigmoid class to apply temperature scaling.
    """

    def __init__(self, t: float = 1.0, s: float = 1.0) -> None:
        """
        Constructor of the Sigmoid class.

        Args:
            t (float): Sigmoid temperature parameter.
            s (float): Sigmoid scaling parameter.
        """
        super().__init__()

        self.t = t
        self.s = s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of the sigmoid layer, following the formula:
        sigmoid(x) = s / (1 + exp(-x / t))

        """
        return self.s / (1 + torch.exp(-x / self.t))


class DecoderBlock(torch.nn.Module):
    """
    Class for the Transformer model.
    """

    def __init__(
        self,
        pitch_dim: int,
        num_heads: int,
        hidden_dim: int,
        ctx_size: int,
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
            MaskedMultiHeadAttention(pitch_dim, num_heads, ctx_size=self.ctx_size)
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
        # if not seq_length:
        #     seq_length = self.pitch_dim

        # Apply LayerNorm
        output: torch.Tensor = self.layer_norm(x)

        # Apply the multi-head attention layers
        output = self.masked_multi_head_attention(output)

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

    def inference(self, x: torch.Tensor, seq_length: int) -> torch.Tensor:
        """
        Computes the forward pass of the decoder.

        Args:
            x (torch.Tensor): Input tensor [batch, ctx_size, pitch_dim].

        Returns:
            torch.Tensor: Output tensor [batch, ctx_size, pitch_dim].
        """
        # if not seq_length:
        #     seq_length = self.pitch_dim

        # Apply LayerNorm
        output: torch.Tensor = self.layer_norm(x)

        # Apply the multi-head attention layers
        output = self.masked_multi_head_attention.inference(output, seq_length)

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


class TransformerDecoder(torch.nn.Module):
    """
    Class for the custom Transformer model.
    """

    def __init__(
        self,
        pitch_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        ctx_size: int,
    ):
        """
        Constructor of the TransformerDecoder class.
        """
        super().__init__()
        self.pitch_dim = pitch_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.ctx_size = ctx_size

        self.positional_encoding: PositionalEncoding = PositionalEncoding(
            pitch_dim=self.pitch_dim, max_len=self.ctx_size
        )
        self.layers = torch.nn.ModuleList(
            [
                DecoderBlock(
                    pitch_dim=self.pitch_dim,
                    num_heads=self.num_heads,
                    hidden_dim=self.hidden_dim,
                    ctx_size=self.ctx_size,
                )
                for _ in range(self.num_layers)
            ]
        )

        # self.linear_out = torch.nn.Linear(pitch_dim, 1)
        self.sigmoid = Sigmoid(t=2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor [batch, ctx_size, pitch_dim].

        Returns:
            torch.Tensor: Output tensor [batch, ctx_size, pitch_dim].
        """
        # if not seq_length:
        #     seq_length = self.pitch_dim

        # Apply the positional encoding
        x = self.positional_encoding(x)

        # Apply the layers
        for layer in self.layers:
            x = layer(x)

        # Apply the sigmoid layer
        x = self.sigmoid(x)

        return x

    def inference(self, x: torch.Tensor = None, seq_length: int = None) -> torch.Tensor:
        """
        Predict the output of the model.

        Args:
            x (torch.Tensor): Input tensor [batch, ctx_size, pitch_dim].

        Returns:
            torch.Tensor: Output tensor [batch, ctx_size, pitch_dim].
        """
        if x is None:
            # x = torch.zeros(1, self.ctx_size, self.pitch_dim).to(torch.device("cuda"))
            x = torch.zeros(1, self.ctx_size, self.pitch_dim).to(torch.device("cpu"))

        if seq_length is None:
            seq_length = 0

        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer.inference(x, seq_length)

        # Apply the sigmoid layer
        x = self.sigmoid(x, seq_length)

        return x
