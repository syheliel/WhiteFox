import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask, inv_scale):
        # Permute the dimensions of the query, key, and value tensors
        q = query.permute(0, 2, 1, 3)  # Shape: [batch_size, seq_len, num_heads, head_dim]
        k = key.permute(0, 2, 1, 3)    # Shape: [batch_size, seq_len, num_heads, head_dim]
        v = value.permute(0, 2, 1, 3)  # Shape: [batch_size, seq_len, num_heads, head_dim]

        # Compute the attention weights
        attn_weights = (torch.matmul(q, k.transpose(-2, -1)).div(inv_scale) + attn_mask).softmax(dim=-1)

        # Apply dropout to the attention weights
        dropout_attn_weights = nn.functional.dropout(attn_weights, self.dropout_p)

        # Multiply the dropout attention weights by the value tensor
        output = dropout_attn_weights.matmul(v)

        return output

# Initialize the model
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
dropout_p = 0.1  # Dropout probability

model = AttentionModel(embed_dim, num_heads, dropout_p)

# Create input tensors for the model
batch_size = 2
seq_len = 10
head_dim = embed_dim // num_heads  # Dimension of each head

query = torch.randn(batch_size, seq_len, num_heads, head_dim)
key = torch.randn(batch_size, seq_len, num_heads, head_dim)
value = torch.randn(batch_size, seq_len, num_heads, head_dim)
attn_mask = torch.zeros(batch_size, num_heads, seq_len, seq_len)  # Attention mask
inv_scale = 1.0 / (head_dim ** 0.5)  # Inverse scale for dot product

# Forward pass through the model
output = model(query, key, value, attn_mask, inv_scale)

# Output shape
print(output.shape)  # Should be [batch_size, seq_len, num_heads, head_dim]
