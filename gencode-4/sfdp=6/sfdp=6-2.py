import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttentionModel(nn.Module):
    def __init__(self, embed_dim, dropout_p):
        super(SelfAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p

    def forward(self, query, key, value):
        q = query.permute(0, 2, 1, 3)  # Permute dimensions of query tensor
        k = key.permute(0, 2, 1, 3)      # Permute dimensions of key tensor
        v = value.permute(0, 2, 1, 3)    # Permute dimensions of value tensor
        
        div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))  # Compute dot product and scale
        div = div.to(torch.float32)  # Convert to float32
        
        attn_weight = F.softmax(div, dim=-1)  # Apply softmax
        attn_weight = F.dropout(attn_weight, p=self.dropout_p, training=self.training)  # Apply dropout
        attn_weight = attn_weight.to(torch.float16)  # Convert to float16
        
        output = attn_weight @ v  # Compute dot product with value
        return output

# Initializing the model
embed_dim = 64  # Example embedding dimension
dropout_p = 0.1  # Dropout probability
model = SelfAttentionModel(embed_dim, dropout_p)

# Inputs to the model
batch_size = 1
seq_length = 10
num_heads = 4
# Example input tensors (query, key, value)
query = torch.randn(batch_size, seq_length, num_heads, embed_dim // num_heads)
key = torch.randn(batch_size, seq_length, num_heads, embed_dim // num_heads)
value = torch.randn(batch_size, seq_length, num_heads, embed_dim // num_heads)

# Forward pass
output = model(query, key, value)

# Display output shape
print(output.shape)  # Expected shape: (1, 10, 4, 16)
