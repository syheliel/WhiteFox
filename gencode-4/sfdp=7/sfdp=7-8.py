import torch
import torch.nn as nn
import math

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_proj(query).permute(0, 2, 1, 3)
        k = self.key_proj(key).permute(0, 2, 1, 3)
        v = self.value_proj(value).permute(0, 2, 1, 3)

        # Compute the dot product of the query and the transposed key, and divide by the square root of the last dimension of the query
        div = (q @ k.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        div = div.to(torch.float32)

        # Apply softmax to the result along the last dimension
        attn_weight = torch.softmax(div, dim=-1)
        attn_weight = attn_weight.to(torch.float16)

        # Compute the dot product of the attention weights and the value
        output = attn_weight @ v
        return output

# Initialize the model
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
model = AttentionModel(embed_dim, num_heads)

# Generate input tensors
batch_size = 2  # Number of sequences in batch
seq_length = 10 # Length of each sequence

# Create random input tensors for query, key, and value
query = torch.randn(batch_size, seq_length, embed_dim)
key = torch.randn(batch_size, seq_length, embed_dim)
value = torch.randn(batch_size, seq_length, embed_dim)

# Get the output from the model
output = model(query, key, value)
print(output.shape)  # Output shape
