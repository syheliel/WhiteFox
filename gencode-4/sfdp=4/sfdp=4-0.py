import torch
import torch.nn as nn
import math

class AttentionModel(nn.Module):
    def __init__(self, embed_size, heads):
        super(AttentionModel, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, attn_mask):
        N = query.shape[0]  # Number of samples
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        # Split into heads
        values = self.values(value).view(N, value_len, self.heads, self.head_dim)
        keys = self.keys(key).view(N, key_len, self.heads, self.head_dim)
        queries = self.queries(query).view(N, query_len, self.heads, self.head_dim)

        # Transpose to get dimensions (N, heads, seq_length, head_dim)
        values = values.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)

        # Compute the dot product attention
        qk = torch.einsum("nqhd,nkhd->nqhk", [queries, keys]) / math.sqrt(self.embed_size)
        qk = qk + attn_mask  # Add the attention mask
        attn_weight = torch.softmax(qk, dim=-1)  # Apply softmax

        output = torch.einsum("nqhk,nvhd->nqhd", [attn_weight, values])  # Compute the output
        output = output.reshape(N, query_len, self.heads * self.head_dim)
        output = self.fc_out(output)  # Pass through the final linear layer

        return output

# Model initialization
embed_size = 64  # Size of the embedding
heads = 8  # Number of attention heads
model = AttentionModel(embed_size, heads)

# Inputs to the model
batch_size = 1
query_len = 10
key_len = 15
value_len = 15
attn_mask = torch.zeros(batch_size, heads, query_len, key_len)  # Attention mask

# Random input tensors
query = torch.randn(batch_size, query_len, embed_size)
key = torch.randn(batch_size, key_len, embed_size)
value = torch.randn(batch_size, value_len, embed_size)

# Forward pass
output = model(query, key, value, attn_mask)

print(output.shape)  # Output shape
