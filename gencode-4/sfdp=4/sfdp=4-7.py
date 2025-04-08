import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionModel(nn.Module):
    def __init__(self, d_model, n_heads):
        super(AttentionModel, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attn_mask=None):
        # Compute the dot product and scale
        qk = (self.query_linear(query) @ self.key_linear(key).transpose(-2, -1)) / math.sqrt(self.d_model)
        
        # Add the attention mask, if provided
        if attn_mask is not None:
            qk = qk + attn_mask
        
        # Apply softmax to get attention weights
        attn_weight = F.softmax(qk, dim=-1)
        
        # Compute the output
        output = attn_weight @ self.value_linear(value)
        return output

# Initializing the model with specific dimensions
d_model = 64  # Dimension of the model
n_heads = 8   # Number of attention heads
attention_model = AttentionModel(d_model=d_model, n_heads=n_heads)

# Inputs to the model
batch_size = 1
seq_length = 10
query = torch.randn(batch_size, seq_length, d_model)  # Query tensor
key = torch.randn(batch_size, seq_length, d_model)    # Key tensor
value = torch.randn(batch_size, seq_length, d_model)  # Value tensor
attn_mask = torch.zeros(batch_size, seq_length, seq_length)  # Attention mask (could be modified as needed)

# Get output from the model
output = attention_model(query, key, value, attn_mask)

print("Output shape:", output.shape)
