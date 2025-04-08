import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.query_layer = nn.Linear(embed_size, embed_size)
        self.key_layer = nn.Linear(embed_size, embed_size)
        self.value_layer = nn.Linear(embed_size, embed_size)
        self.num_heads = num_heads
        self.scale = embed_size ** 0.5

    def forward(self, query, key, value):
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(query, key.transpose(-2, -1))  # Shape: (batch_size, num_heads, seq_length, seq_length)

        # Scale the dot product by the inverse scale
        scaled_qk = qk / self.scale  # Shape: (batch_size, num_heads, seq_length, seq_length)

        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)  # Shape: (batch_size, num_heads, seq_length, seq_length)

        # Compute the dot product of the softmax output and the value tensor
        output = softmax_qk.matmul(value)  # Shape: (batch_size, num_heads, seq_length, embed_size)
        
        return output

# Initializing the model
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads
model = AttentionModel(embed_size, num_heads)

# Inputs to the model
batch_size = 1
seq_length = 10
query = torch.randn(batch_size, num_heads, seq_length, embed_size)  # Shape: (1, 8, 10, 64)
key = torch.randn(batch_size, num_heads, seq_length, embed_size)    # Shape: (1, 8, 10, 64)
value = torch.randn(batch_size, num_heads, seq_length, embed_size)  # Shape: (1, 8, 10, 64)

# Forward pass
output = model(query, key, value)

print(output.shape)  # Output shape: (1, 8, 10, 64)
