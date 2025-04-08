import torch
import torch.nn as nn
import math

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.embed_dim = embed_dim

    def forward(self, x, attn_mask):
        # Compute the query, key, and value
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Compute the scaled dot-product attention
        qk = q @ k.transpose(-2, -1) / math.sqrt(self.embed_dim)
        qk = qk + attn_mask  # Add the attention mask
        attn_weight = torch.softmax(qk, dim=-1)  # Apply softmax
        attn_weight = self.dropout(attn_weight)  # Apply dropout
        output = attn_weight @ v  # Compute the output

        return output

# Initializing the model
embed_dim = 64  # Example embedding dimension
num_heads = 8   # Example number of attention heads
dropout_p = 0.1 # Dropout probability
model = AttentionModel(embed_dim, num_heads, dropout_p)

# Inputs to the model
batch_size = 1
seq_length = 10
x = torch.randn(batch_size, seq_length, embed_dim)  # Example input tensor
attn_mask = torch.zeros(batch_size, seq_length, seq_length)  # Example attention mask (no masking)

# Forward pass through the model
output = model(x, attn_mask)

# Output shape
print(output.shape)  # Should be (batch_size, seq_length, embed_dim)
