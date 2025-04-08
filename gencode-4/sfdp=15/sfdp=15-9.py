import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, attn_mask=None, inv_scale=1.0):
        q = self.query_linear(query).permute(0, 2, 1, 3) # Shape: (batch_size, num_heads, seq_len, embed_size/num_heads)
        k = self.key_linear(key).permute(0, 2, 1, 3)     # Shape: (batch_size, num_heads, seq_len, embed_size/num_heads)
        v = self.value_linear(value).permute(0, 2, 1, 3) # Shape: (batch_size, num_heads, seq_len, embed_size/num_heads)

        attn_weights = (torch.matmul(q, k.transpose(-2, -1)).div(inv_scale) + attn_mask).softmax(dim=-1) # Compute attention weights
        dropout_attn_weights = nn.functional.dropout(attn_weights, p=self.dropout_p) # Apply dropout to attention weights
        output = dropout_attn_weights.matmul(v) # Multiply by value tensor
        
        return output

# Initialize model parameters
embed_size = 64  # Embedding size
num_heads = 8    # Number of attention heads
dropout_p = 0.1  # Dropout probability

# Instantiate the model
model = AttentionModel(embed_size, num_heads, dropout_p)

# Generate input tensors
batch_size = 1
seq_len = 10  # Sequence length
query = torch.randn(batch_size, seq_len, embed_size)  # Query tensor
key = torch.randn(batch_size, seq_len, embed_size)    # Key tensor
value = torch.randn(batch_size, seq_len, embed_size)  # Value tensor
attn_mask = torch.zeros(batch_size, seq_len, seq_len) # Attention mask (can be modified based on your scenario)

# Forward pass
output = model(query, key, value, attn_mask)

# Output tensor
print(output)
