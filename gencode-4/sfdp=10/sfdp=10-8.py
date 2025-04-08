import torch
import torch.nn as nn

# Model Definition
class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(ScaledDotProductAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by heads"
        
    def forward(self, query, key, value, inv_scale):
        # Permute the dimensions of the query, key, and value tensors
        q = query.permute(0, 2, 1, 3)  # (batch_size, seq_length, heads, head_dim)
        k = key.permute(0, 2, 1, 3)    # (batch_size, seq_length, heads, head_dim)
        v = value.permute(0, 2, 1, 3)  # (batch_size, seq_length, heads, head_dim)
        
        # Compute the dot product of the query and the transposed key
        attention = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, heads, seq_length, seq_length)
        
        # Scale the attention by dividing it by the inverse scale
        scaled_attention = attention.div(inv_scale)
        
        # Apply softmax to the scaled attention
        attention_weights = scaled_attention.softmax(dim=-1)  # (batch_size, heads, seq_length, seq_length)
        
        # Compute the weighted sum of the value tensor
        output = attention_weights.matmul(v)  # (batch_size, heads, seq_length, head_dim)
        
        return output.permute(0, 2, 1, 3)  # Permute back to (batch_size, seq_length, heads, head_dim)

# Initializing the model
embed_size = 64  # Embedding size (must be divisible by heads)
heads = 8       # Number of attention heads
model = ScaledDotProductAttention(embed_size, heads)

# Inputs to the model
batch_size = 2
seq_length = 10
inv_scale = (embed_size ** 0.5)  # Example inverse scale
query = torch.randn(batch_size, seq_length, embed_size)
key = torch.randn(batch_size, seq_length, embed_size)
value = torch.randn(batch_size, seq_length, embed_size)

# Forward pass through the model
output = model(query, key, value, inv_scale)

# Output shape
print(output.shape)  # Expected shape: (batch_size, seq_length, heads, head_dim)
