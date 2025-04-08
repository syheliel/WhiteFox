import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super(SelfAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        
        # Define linear layers for query, key, and value
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_layer(query).permute(0, 2, 1, 3)  # Shape: (batch_size, num_heads, seq_len_q, head_dim)
        k = self.key_layer(key).permute(0, 2, 1, 3)      # Shape: (batch_size, num_heads, seq_len_k, head_dim)
        v = self.value_layer(value).permute(0, 2, 1, 3)  # Shape: (batch_size, num_heads, seq_len_v, head_dim)
        
        # Compute the dot product and scale
        div = (q @ k.transpose(-2, -1)) / math.sqrt(self.embed_dim // self.num_heads)
        div = div.to(torch.float32)  # Convert to float32
        
        # Apply softmax and dropout
        attn_weight = F.softmax(div, dim=-1)
        attn_weight = self.dropout(attn_weight)  # Apply dropout
        attn_weight = attn_weight.to(torch.float16)  # Convert to float16
        
        # Compute the dot product of attention weights and values
        output = attn_weight @ v  # Shape: (batch_size, num_heads, seq_len_q, head_dim)
        return output

# Initializing the model
embed_dim = 64   # Embedding dimension
num_heads = 8    # Number of attention heads
dropout_p = 0.1  # Dropout rate
model = SelfAttentionModel(embed_dim, num_heads, dropout_p)

# Inputs to the model
batch_size = 1
seq_len = 10  # Sequence length
input_tensor = torch.randn(batch_size, seq_len, embed_dim)  # Shape: (batch_size, seq_len, embed_dim)

# Example usage
output = model(input_tensor, input_tensor, input_tensor)  # Using the same tensor for query, key, and value
print(output.shape)  # Should be: (batch_size, num_heads, seq_len, head_dim)
