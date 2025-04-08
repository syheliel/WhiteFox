import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super().__init__()
        self.query_projection = nn.Linear(embed_size, embed_size)
        self.key_projection = nn.Linear(embed_size, embed_size)
        self.value_projection = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.embed_size = embed_size
        self.num_heads = num_heads

    def forward(self, query, key, value, attn_mask=None):
        # Compute the scaled dot product attention
        qk = self.query_projection(query) @ self.key_projection(key).transpose(-2, -1) / (self.embed_size ** 0.5)
        
        if attn_mask is not None:
            qk = qk + attn_mask  # Add the attention mask
        
        attn_weight = F.softmax(qk, dim=-1)  # Apply softmax to the result
        attn_weight = self.dropout(attn_weight)  # Apply dropout to the softmax output
        
        output = attn_weight @ self.value_projection(value)  # Compute the output
        return output

# Initializing the model
embed_size = 64  # Size of each embedding vector
num_heads = 8    # Number of attention heads
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_size, num_heads, dropout_p)

# Inputs to the model
batch_size = 1
seq_length = 10  # Length of the sequence
query = torch.randn(batch_size, seq_length, embed_size)  # Query tensor
key = torch.randn(batch_size, seq_length, embed_size)    # Key tensor
value = torch.randn(batch_size, seq_length, embed_size)  # Value tensor
attn_mask = torch.zeros(batch_size, seq_length, seq_length)  # Attention mask (optional)

# Forward pass
output = model(query, key, value, attn_mask)

# Output shape
print(output.shape)  # Should be (1, 10, 64) as per the input dimensions
