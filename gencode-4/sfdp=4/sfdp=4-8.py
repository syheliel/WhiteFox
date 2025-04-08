import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionModel(nn.Module):
    def __init__(self, embed_size, heads):
        super(AttentionModel, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size should be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x, attn_mask):
        N, seq_length, _ = x.shape
        
        # Split the embedding into multiple heads
        values = self.values(x).view(N, seq_length, self.heads, self.head_dim)
        keys = self.keys(x).view(N, seq_length, self.heads, self.head_dim)
        queries = self.queries(x).view(N, seq_length, self.heads, self.head_dim)
        
        # Transpose to get dimensions N * heads * seq_length * head_dim
        values = values.transpose(1, 2)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # Compute the dot product of the query and key, and scale it
        qk = torch.einsum("nhqd,nhkd->nhqk", [queries, keys]) / math.sqrt(self.embed_size)
        qk = qk + attn_mask  # Add the attention mask

        # Apply softmax to the result
        attn_weight = F.softmax(qk, dim=-1)

        # Compute the dot product of the attention weights and the value
        output = torch.einsum("nhqk,nhvd->nhqd", [attn_weight, values]).reshape(N, seq_length, self.embed_size)

        return self.fc_out(output)

# Initializing the model
embed_size = 64  # Size of the embedding
heads = 8  # Number of attention heads
model = AttentionModel(embed_size, heads)

# Inputs to the model
N, seq_length = 1, 10  # Batch size and sequence length
x = torch.randn(N, seq_length, embed_size)  # Input tensor
attn_mask = torch.zeros(N, heads, seq_length, seq_length)  # Attention mask (for simplicity, a zero mask)

# Forward pass
output = model(x, attn_mask)

print(output.shape)  # Should output (N, seq_length, embed_size)
