import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, query, key, value, inv_scale):
        q = query.permute(0, 2, 1, 3)  # Permute the dimensions of the query tensor
        k = key.permute(0, 2, 1, 3)      # Permute the dimensions of the key tensor
        v = value.permute(0, 2, 1, 3)    # Permute the dimensions of the value tensor

        attention = torch.matmul(q, k.transpose(-2, -1))  # Compute dot product
        scaled_attention = attention.div(inv_scale)         # Scale the attention
        attention_weights = scaled_attention.softmax(dim=-1) # Apply softmax
        output = attention_weights.matmul(v)                 # Compute weighted sum of the value tensor

        return output

# Initialize the model
embed_dim = 64  # Example embedding dimension
model = ScaledDotProductAttention(embed_dim)

# Inputs to the model
batch_size = 2   # Example batch size
seq_length = 10  # Example sequence length
num_heads = 4    # Example number of attention heads

# Create random input tensors for query, key, and value
query = torch.randn(batch_size, seq_length, num_heads, embed_dim)
key = torch.randn(batch_size, seq_length, num_heads, embed_dim)
value = torch.randn(batch_size, seq_length, num_heads, embed_dim)
inv_scale = embed_dim ** 0.5  # Inverse scale (typically sqrt(d_k))

# Forward pass through the model
output = model(query, key, value, inv_scale)

print(output.shape)  # Output shape should be (batch_size, seq_length, num_heads, embed_dim)
