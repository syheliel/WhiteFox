import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, query, key, value, attn_mask):
        N, value_len, _ = value.size()
        N, query_len, _ = query.size()
        
        # Compute the dot product of the query and key, and scale it
        qk = (query @ key.transpose(-2, -1)) / math.sqrt(self.embed_size)

        # Add the attention mask to the scaled dot product
        qk = qk + attn_mask
        
        # Apply softmax to the result
        attn_weight = F.softmax(qk, dim=-1)
        
        # Compute the dot product of the attention weights and the value
        output = attn_weight @ value
        
        return self.fc_out(output)

# Initialize the model
embed_size = 64  # Embedding size
num_heads = 8    # Number of attention heads
model = AttentionModel(embed_size, num_heads)

# Inputs to the model
batch_size = 1
query_len = 10
key_len = 10
value_len = 10

# Random inputs
query = torch.randn(batch_size, query_len, embed_size)
key = torch.randn(batch_size, key_len, embed_size)
value = torch.randn(batch_size, value_len, embed_size)

# Attention mask (for simplicity, using a mask with all zeros)
attn_mask = torch.zeros(batch_size, query_len, key_len)

# Forward pass
output = model(query, key, value, attn_mask)

print(output.shape)  # Output shape
