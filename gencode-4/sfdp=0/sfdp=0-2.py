import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.embed_dim = embed_dim
        self.inv_scale = 1.0 / (embed_dim ** 0.5)

    def forward(self, query, key, value):
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(query, key.transpose(-2, -1))
        # Scale the dot product by the inverse scale
        scaled_qk = qk * self.inv_scale
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        # Compute the dot product of the softmax output and the value tensor
        output = softmax_qk.matmul(value)
        return output

# Initialize the model with an embedding dimension
embed_dim = 64  # Example embedding dimension
model = ScaledDotProductAttention(embed_dim)

# Inputs to the model
# Example tensors for query, key, and value
batch_size = 2  # Example batch size
num_heads = 4   # Number of attention heads
seq_length = 10 # Sequence length

# Create random input tensors for query, key, and value
query = torch.randn(batch_size, num_heads, seq_length, embed_dim)
key = torch.randn(batch_size, num_heads, seq_length, embed_dim)
value = torch.randn(batch_size, num_heads, seq_length, embed_dim)

# Forward pass through the model
output = model(query, key, value)

# Print the output shape
print(output.shape)  # Expected output shape: (batch_size, num_heads, seq_length, embed_dim)
