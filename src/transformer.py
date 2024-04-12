import torch
import torch.nn.functional as F


class AttentionHead(torch.nn.Module):
    def __init__(self, embed_dim: int, head_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim

        self.q = torch.nn.Linear(embed_dim, head_dim)
        self.k = torch.nn.Linear(embed_dim, head_dim)
        self.v = torch.nn.Linear(embed_dim, head_dim)

    def forward(self, x: torch.Tensor):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        dim_k: int = k.shape[-1]
        matmul = torch.matmul(q, k.T)
        scaled = matmul / (dim_k**0.5)
        softmax = F.softmax(scaled, dim=-1)
        output = torch.matmul(softmax, v)
        return output


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mask: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = embed_dim // num_heads

        self.heads = torch.nn.ModuleList(
            [AttentionHead(embed_dim, self.head_dim) for _ in range(num_heads)]
        )
        self.output = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        heads = [head(x) for head in self.heads]
        output = torch.cat(heads, dim=-1)
        output = self.output(output)
        return output


class Transformer(torch.autograd.Function):
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
        # self.attention = AttentionHead(512, 64)
        # self.transformer = Transformer(512, 512, 8, 6, 2048, dropout=0.1)
        self.outputs = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        pass
