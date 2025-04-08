import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(ScaledDotProductAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by heads"

    def forward(self, query, key, value):
        # Permute the dimensions of the query, key, and value tensors
        q = query.permute(0, 2, 1, 3)  # (batch_size, seq_length, heads, head_dim)
        k = key.permute(0, 2, 1, 3)    # (batch_size, seq_length, heads, head_dim)
        v = value.permute(0, 2, 1, 3)  # (batch_size, seq_length, heads, head_dim)

        # Compute the dot product of the query and the transposed key
        attention = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, heads, seq_length, seq_length)
        
        inv_scale = self.head_dim ** 0.5  # Scaling factor
        scaled_attention = attention / inv_scale  # Scale the attention
        
        # Apply softmax to the scaled attention
        attention_weights = scaled_attention.softmax(dim=-1)  # (batch_size, heads, seq_length, seq_length)
        
        # Compute the weighted sum of the value tensor
        output = attention_weights.matmul(v)  # (batch_size, heads, seq_length, head_dim)
        
        return output.permute(0, 2, 1, 3).contiguous()  # Permute back to (batch_size, seq_length, heads, head_dim)

# Initialize the model
embed_size = 64  # Size of each embedding
heads = 8       # Number of attention heads
model = ScaledDotProductAttention(embed_size, heads)

# Inputs to the model
batch_size = 1
seq_length = 10  # Length of the input sequence
query = torch.randn(batch_size, seq_length, heads, embed_size // heads)
key = torch.randn(batch_size, seq_length, heads, embed_size // heads)
value = torch.randn(batch_size, seq_length, heads, embed_size // heads)

# Forward pass through the model
output = model(query, key, value)

print("Output shape:", output.shape)  # Output shape should be (batch_size, seq_length, heads, head_dim)
