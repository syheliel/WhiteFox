import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super().__init__()
        self.query_layer = nn.Linear(embed_size, embed_size)
        self.key_layer = nn.Linear(embed_size, embed_size)
        self.value_layer = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.embed_size = embed_size
        self.num_heads = num_heads

    def forward(self, query, key, value, attn_mask):
        # Compute the scaled dot product attention
        qk = (self.query_layer(query) @ self.key_layer(key).transpose(-2, -1)) / (self.embed_size ** 0.5)  # Scale
        qk = qk + attn_mask  # Add the attention mask
        attn_weight = F.softmax(qk, dim=-1)  # Softmax
        attn_weight = self.dropout(attn_weight)  # Dropout
        output = attn_weight @ self.value_layer(value)  # Output
        return output

# Model initialization
embed_size = 64  # Embedding size
num_heads = 8    # Number of attention heads
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_size, num_heads, dropout_p)

# Generating input tensors
batch_size = 1
seq_length = 10  # Length of the input sequence

query = torch.randn(batch_size, seq_length, embed_size)
key = torch.randn(batch_size, seq_length, embed_size)
value = torch.randn(batch_size, seq_length, embed_size)
attn_mask = torch.zeros(batch_size, seq_length, seq_length)  # Assuming no mask for simplicity

# Forward pass
output = model(query, key, value, attn_mask)

# Output shape
print(output.shape)  # Should be (batch_size, seq_length, embed_size)
