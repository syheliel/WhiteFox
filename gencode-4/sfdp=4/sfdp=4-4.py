import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(ScaledDotProductAttention, self).__init__()
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
        length = query.shape[1]  # Sequence length
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        # Split the embedding into multiple heads
        queries = self.queries(query).view(N, query_len, self.heads, self.head_dim).transpose(1, 2)
        keys = self.keys(key).view(N, key_len, self.heads, self.head_dim).transpose(1, 2)
        values = self.values(value).view(N, value_len, self.heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        qk = (queries @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        qk = qk + attn_mask  # Add attention mask
        attn_weight = torch.softmax(qk, dim=-1)  # Softmax over the last dimension
        output = attn_weight @ values  # Dot product with values

        # Concatenate heads and pass through final linear layer
        output = output.transpose(1, 2).contiguous().view(N, query_len, self.embed_size)
        return self.fc_out(output)

# Initializing the model
embed_size = 64  # Size of the embedding
heads = 8  # Number of attention heads
model = ScaledDotProductAttention(embed_size=embed_size, heads=heads)

# Inputs to the model
batch_size = 1
seq_length = 10  # Length of the input sequences
query = torch.randn(batch_size, seq_length, embed_size)
key = torch.randn(batch_size, seq_length, embed_size)
value = torch.randn(batch_size, seq_length, embed_size)
attn_mask = torch.zeros(batch_size, heads, seq_length, seq_length)  # Example attention mask (no masking)

# Output from the model
output = model(query, key, value, attn_mask)

# Output shape
print(output.shape)  # Should be (batch_size, seq_length, embed_size)
