import torch
import torch.nn as nn
import math

class SelfAttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super(SelfAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value):
        # Permute the dimensions
        q = self.query_linear(query).permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)
        k = self.key_linear(key).permute(0, 2, 1, 3)      # (batch_size, seq_len, num_heads, head_dim)
        v = self.value_linear(value).permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)

        # Scale the query tensor
        q = q / math.sqrt(q.size(-1))

        # Matrix multiplication
        div = q @ k.transpose(-2, -1)  # (batch_size, num_heads, seq_len, seq_len)

        # Convert to float32
        div = div.to(torch.float32)

        # Apply softmax
        attn_weight = torch.softmax(div, dim=-1)

        # Apply dropout
        attn_weight = self.dropout(attn_weight)

        # Convert to float16
        attn_weight = attn_weight.to(torch.float16)

        # Matrix multiplication with value tensor
        output = attn_weight @ v  # (batch_size, num_heads, seq_len, head_dim)

        return output

# Initialize the model
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
dropout_p = 0.1 # Dropout probability
model = SelfAttentionModel(embed_dim, num_heads, dropout_p)

# Generate input tensors
batch_size = 2
seq_len = 10
input_tensor = torch.randn(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)

# Forward pass through the model
output = model(input_tensor, input_tensor, input_tensor)

print("Output shape:", output.shape)
