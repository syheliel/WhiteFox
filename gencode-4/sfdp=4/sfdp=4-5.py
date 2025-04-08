import torch
import torch.nn as nn
import math

class AttentionModel(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
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
        N = query.shape[0]  # Batch size
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        # Split embedding into multiple heads
        queries = self.queries(query).view(N, query_len, self.heads, self.head_dim).transpose(1, 2)
        keys = self.keys(key).view(N, key_len, self.heads, self.head_dim).transpose(1, 2)
        values = self.values(value).view(N, value_len, self.heads, self.head_dim).transpose(1, 2)

        # Compute the dot product of the query and key, and scale it
        qk = (queries @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        qk = qk + attn_mask  # Add the attention mask

        # Apply softmax to the result
        attn_weight = torch.softmax(qk, dim=-1)

        # Compute the dot product of the attention weights and the value
        output = attn_weight @ values
        output = output.transpose(1, 2).contiguous().view(N, query_len, self.embed_size)
        return self.fc_out(output)

# Model initialization
embed_size = 64  # Embedding size
heads = 8  # Number of attention heads
model = AttentionModel(embed_size, heads)

# Generating random input tensors
batch_size = 1
query_len = 10
key_len = 15
value_len = 15

query = torch.randn(batch_size, query_len, embed_size)
key = torch.randn(batch_size, key_len, embed_size)
value = torch.randn(batch_size, value_len, embed_size)
attn_mask = torch.zeros(batch_size, heads, query_len, key_len)  # Attention mask

# Output from the model
output = model(query, key, value, attn_mask)

print(output.shape)  # Expected output shape: (batch_size, query_len, embed_size)
