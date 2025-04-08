import torch
import torch.nn as nn
import math

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        
    def forward(self, query, key, value, attn_mask):
        # Compute the dot product of the query and key, and scale it
        qk = (self.query_linear(query) @ self.key_linear(key).transpose(-2, -1)) / math.sqrt(self.embed_size)
        # Add the attention mask to the scaled dot product
        qk = qk + attn_mask
        # Apply softmax to the result
        attn_weight = torch.softmax(qk, dim=-1)
        # Compute the dot product of the attention weights and the value
        output = attn_weight @ self.value_linear(value)
        return output

# Initializing the model with embedding size 64 and 1 head
model = AttentionModel(embed_size=64, num_heads=1)

# Generating input tensors
batch_size = 2  # Number of samples in the batch
seq_length = 10  # Length of the input sequences

# Random input tensors for query, key, value, and attention mask
query = torch.randn(batch_size, seq_length, 64)
key = torch.randn(batch_size, seq_length, 64)
value = torch.randn(batch_size, seq_length, 64)
attn_mask = torch.zeros(batch_size, seq_length, seq_length)  # Example of a zero mask (no masking)

# Forward pass through the model
output = model(query, key, value, attn_mask)

# Display the output
print(output)
