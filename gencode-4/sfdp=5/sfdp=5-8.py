import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super().__init__()
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.embed_size = embed_size
        self.num_heads = num_heads

    def forward(self, query, key, value, attn_mask):
        # Compute the dot product of the query and key, and scale it
        qk = self.query_linear(query) @ self.key_linear(key).transpose(-2, -1) / math.sqrt(self.embed_size)
        
        # Add the attention mask to the scaled dot product
        qk = qk + attn_mask
        
        # Apply softmax to the result
        attn_weight = F.softmax(qk, dim=-1)
        
        # Apply dropout to the softmax output
        attn_weight = self.dropout(attn_weight)
        
        # Compute the dot product of the dropout output and the value
        output = attn_weight @ self.value_linear(value)
        
        return output

# Initializing the model
embed_size = 64  # Size of each embedding
num_heads = 8    # Number of attention heads
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_size, num_heads, dropout_p)

# Inputs to the model
batch_size = 1
seq_length = 10  # Length of the input sequences

# Random input tensors for query, key, and value
query = torch.randn(batch_size, seq_length, embed_size)
key = torch.randn(batch_size, seq_length, embed_size)
value = torch.randn(batch_size, seq_length, embed_size)

# Attention mask (should be of shape [batch_size, seq_length, seq_length])
attn_mask = torch.zeros(batch_size, seq_length, seq_length)  # No masking in this example

# Get the output from the model
output = model(query, key, value, attn_mask)

print("Output shape:", output.shape)  # Should be [batch_size, seq_length, embed_size]
