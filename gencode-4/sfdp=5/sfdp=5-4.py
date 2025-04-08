import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super().__init__()
        self.query_proj = nn.Linear(embed_size, embed_size)
        self.key_proj = nn.Linear(embed_size, embed_size)
        self.value_proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.embed_size = embed_size
        self.num_heads = num_heads

    def forward(self, query, key, value, attn_mask):
        # Compute the dot product of the query and key, and scale it
        qk = (self.query_proj(query) @ self.key_proj(key).transpose(-2, -1)) / (self.embed_size ** 0.5)
        
        # Add the attention mask to the scaled dot product
        qk = qk + attn_mask
        
        # Apply softmax to the result
        attn_weight = F.softmax(qk, dim=-1)
        
        # Apply dropout to the softmax output
        attn_weight = self.dropout(attn_weight)

        # Compute the dot product of the dropout output and the value
        output = attn_weight @ self.value_proj(value)
        
        return output

# Initializing the model parameters
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads
dropout_p = 0.1  # Dropout probability

# Create an instance of the model
model = AttentionModel(embed_size, num_heads, dropout_p)

# Inputs to the model (batch_size, seq_length, embed_size)
batch_size = 4
seq_length = 10
query = torch.randn(batch_size, seq_length, embed_size)
key = torch.randn(batch_size, seq_length, embed_size)
value = torch.randn(batch_size, seq_length, embed_size)

# Attention mask (batch_size, seq_length, seq_length)
attn_mask = torch.zeros(batch_size, seq_length, seq_length)  # No masking for this example

# Forward pass through the model
output = model(query, key, value, attn_mask)

print(output.shape)  # Output shape should be (batch_size, seq_length, embed_size)
