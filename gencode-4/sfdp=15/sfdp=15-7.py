import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value, attn_mask=None, inv_scale=1.0):
        q = self.query(query).permute(0, 2, 1, 3)  # Permute query tensor
        k = self.key(key).permute(0, 2, 1, 3)      # Permute key tensor
        v = self.value(value).permute(0, 2, 1, 3)  # Permute value tensor

        attn_weights = (torch.matmul(q, k.transpose(-2, -1)).div(inv_scale) + attn_mask).softmax(dim=-1)  # Compute attention weights
        dropout_attn_weights = self.dropout(attn_weights)  # Apply dropout to attention weights
        output = dropout_attn_weights.matmul(v)  # Multiply by value tensor
        return output

# Initializing the model
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
dropout_p = 0.1 # Dropout probability

model = AttentionModel(embed_dim=embed_dim, num_heads=num_heads, dropout_p=dropout_p)

# Create input tensors
batch_size = 1
seq_length = 10  # Sequence length
num_values = 5   # Number of values

# Random tensors for query, key, and value
query_tensor = torch.randn(batch_size, seq_length, num_values, embed_dim)
key_tensor = torch.randn(batch_size, seq_length, num_values, embed_dim)
value_tensor = torch.randn(batch_size, seq_length, num_values, embed_dim)

# Attention mask (optional)
attn_mask = torch.zeros(batch_size, seq_length, seq_length).to(torch.float32)

# Inverse scale factor
inv_scale = embed_dim ** 0.5

# Forward pass
output = model(query_tensor, key_tensor, value_tensor, attn_mask=attn_mask, inv_scale=inv_scale)
