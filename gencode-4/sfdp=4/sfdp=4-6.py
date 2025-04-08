import torch
import torch.nn as nn
import math

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(AttentionModel, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size must be divisible by num_heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x, attn_mask):
        N, seq_length, _ = x.shape

        # Linear transformations
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        # Reshape for multi-head attention
        values = values.view(N, seq_length, self.num_heads, self.head_dim)
        keys = keys.view(N, seq_length, self.num_heads, self.head_dim)
        queries = queries.view(N, seq_length, self.num_heads, self.head_dim)

        # Transpose for dot product
        values = values.permute(0, 2, 1, 3)  # (N, num_heads, seq_length, head_dim)
        keys = keys.permute(0, 2, 1, 3)      # (N, num_heads, seq_length, head_dim)
        queries = queries.permute(0, 2, 1, 3) # (N, num_heads, seq_length, head_dim)

        # Compute the dot product and scale
        qk = torch.einsum("nhqd,nhkd->nhqk", queries, keys) / math.sqrt(self.head_dim)
        qk = qk + attn_mask  # Add attention mask

        # Apply softmax to get attention weights
        attn_weights = torch.softmax(qk, dim=-1)

        # Compute the output
        output = torch.einsum("nhql,nhld->nhqd", attn_weights, values)
        output = output.permute(0, 2, 1, 3).contiguous()  # (N, seq_length, num_heads, head_dim)
        output = output.view(N, seq_length, self.embed_size)  # (N, seq_length, embed_size)

        return self.fc_out(output)

# Initialize model
embed_size = 64  # Embedding size
num_heads = 8    # Number of heads
model = AttentionModel(embed_size, num_heads)

# Inputs to the model
batch_size = 2
seq_length = 10
x = torch.randn(batch_size, seq_length, embed_size)  # Input tensor
attn_mask = torch.zeros(batch_size, num_heads, seq_length, seq_length)  # Attention mask

# Forward pass
output = model(x, attn_mask)

# Output
print(output.shape)  # Should be (batch_size, seq_length, embed_size)
