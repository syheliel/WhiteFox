import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttentionModel(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout_p):
        super(SelfAttentionModel, self).__init__()
        self.query = nn.Linear(input_dim, embed_dim)
        self.key = nn.Linear(input_dim, embed_dim)
        self.value = nn.Linear(input_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # Compute query, key, and value
        q = self.query(x).permute(0, 2, 1)  # Shape: (batch_size, embed_dim, seq_len)
        k = self.key(x).permute(0, 2, 1)    # Shape: (batch_size, embed_dim, seq_len)
        v = self.value(x).permute(0, 2, 1)  # Shape: (batch_size, embed_dim, seq_len)

        # Compute the attention scores
        div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))  # Shape: (batch_size, embed_dim, embed_dim)
        div = div.to(torch.float32)  # Convert to float32
        attn_weight = F.softmax(div, dim=-1)  # Apply softmax
        attn_weight = self.dropout(attn_weight)  # Apply dropout
        attn_weight = attn_weight.to(torch.float16)  # Convert to float16

        # Compute the attention output
        output = attn_weight @ v  # Shape: (batch_size, embed_dim, seq_len)
        return output.permute(0, 2, 1)  # Return to original shape (batch_size, seq_len, embed_dim)

# Initializing the model
input_dim = 64  # Input feature dimension
embed_dim = 32  # Embedding dimension for attention
dropout_p = 0.1  # Dropout probability

model = SelfAttentionModel(input_dim=input_dim, embed_dim=embed_dim, dropout_p=dropout_p)

# Inputs to the model
batch_size = 1
seq_length = 10
x = torch.randn(batch_size, seq_length, input_dim)  # Shape: (batch_size, seq_length, input_dim)
output = model(x)

print("Output shape:", output.shape)  # Should be (batch_size, seq_length, embed_dim)
