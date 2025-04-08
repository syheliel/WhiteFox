import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value, inv_scale, attn_mask):
        q = self.query_projection(query).permute(0, 2, 1, 3)  # Permute the dimensions of the query tensor
        k = self.key_projection(key).permute(0, 2, 1, 3)      # Permute the dimensions of the key tensor
        v = self.value_projection(value).permute(0, 2, 1, 3)  # Permute the dimensions of the value tensor
        
        attn_weights = (torch.matmul(q, k.transpose(-2, -1)).div(inv_scale) + attn_mask).softmax(dim=-1)  # Compute the attention weights
        dropout_attn_weights = self.dropout(attn_weights)  # Apply dropout to the attention weights
        output = dropout_attn_weights.matmul(v)  # Multiply the dropout attention weights by the value tensor
        
        return output

# Initialize the model
embed_dim = 64  # Example embedding dimension
num_heads = 8   # Number of attention heads
dropout_p = 0.1 # Dropout probability
model = AttentionModel(embed_dim, num_heads, dropout_p)

# Create input tensors
batch_size = 2
seq_length = 10

query = torch.randn(batch_size, seq_length, num_heads, embed_dim // num_heads)
key = torch.randn(batch_size, seq_length, num_heads, embed_dim // num_heads)
value = torch.randn(batch_size, seq_length, num_heads, embed_dim // num_heads)

# Scale and mask
inv_scale = torch.tensor(1.0)  # Inverse scale for attention scores
attn_mask = torch.zeros(batch_size, num_heads, seq_length, seq_length)  # Attention mask (can be adjusted as needed)

# Forward pass through the model
output = model(query, key, value, inv_scale, attn_mask)

print(output.shape)  # Output shape
