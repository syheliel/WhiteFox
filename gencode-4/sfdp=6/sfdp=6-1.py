import torch
import torch.nn as nn
import math

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # Assume x has shape (batch_size, seq_length, embed_dim)
        q = self.query(x).permute(0, 2, 1)  # (batch_size, embed_dim, seq_length)
        k = self.key(x).permute(0, 2, 1)    # (batch_size, embed_dim, seq_length)
        v = self.value(x).permute(0, 2, 1)  # (batch_size, embed_dim, seq_length)

        # Compute the dot product and scale
        div = q @ k.transpose(-2, -1) / math.sqrt(self.embed_dim)  # (batch_size, seq_length, seq_length)
        div = div.to(torch.float32)  # Convert to float32

        # Apply softmax and dropout
        attn_weight = torch.softmax(div, dim=-1)  # (batch_size, seq_length, seq_length)
        attn_weight = self.dropout(attn_weight)  # Apply dropout
        attn_weight = attn_weight.to(torch.float16)  # Convert to float16

        # Compute the attention output
        output = attn_weight @ v  # (batch_size, seq_length, embed_dim)
        return output

# Initializing the model
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
dropout_p = 0.1 # Dropout probability
model = AttentionModel(embed_dim, num_heads, dropout_p)

# Input tensor
batch_size = 1
seq_length = 10
input_tensor = torch.randn(batch_size, seq_length, embed_dim)

# Forward pass through the model
output = model(input_tensor)

print("Output shape:", output.shape)
