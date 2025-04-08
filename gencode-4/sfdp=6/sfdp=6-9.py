import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, dropout_p):
        super(AttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p

    def forward(self, query, key, value):
        # Permute the dimensions of the query, key, and value tensors
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)

        # Compute the dot product and scale it
        div = q @ k.transpose(-2, -1) / math.sqrt(self.embed_dim)

        # Convert to float32 for stability
        div = div.to(torch.float32)

        # Apply softmax to get attention weights
        attn_weight = F.softmax(div, dim=-1)

        # Apply dropout
        attn_weight = F.dropout(attn_weight, p=self.dropout_p, training=self.training)

        # Convert attention weights to float16
        attn_weight = attn_weight.to(torch.float16)

        # Compute the final attention output
        output = attn_weight @ v
        return output

# Initialize the model
embed_dim = 64  # Example embedding dimension
dropout_p = 0.1  # Example dropout probability
model = AttentionModel(embed_dim, dropout_p)

# Inputs to the model
batch_size = 1
seq_length = 10
num_heads = 4  # Example number of heads
input_tensor_shape = (batch_size, seq_length, num_heads, embed_dim)

query = torch.randn(input_tensor_shape)  # Shape: (1, 10, 4, 64)
key = torch.randn(input_tensor_shape)    # Shape: (1, 10, 4, 64)
value = torch.randn(input_tensor_shape)  # Shape: (1, 10, 4, 64)

# Forward pass through the model
__output__ = model(query, key, value)
