import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.scale = embed_size ** 0.5
        
        # Define linear layers for query, key, and value
        self.query_layer = nn.Linear(embed_size, embed_size)
        self.key_layer = nn.Linear(embed_size, embed_size)
        self.value_layer = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value):
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale the dot product by the inverse scale
        scaled_qk = qk / self.scale
        
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        
        # Compute the dot product of the softmax output and the value tensor
        output = softmax_qk.matmul(value)
        
        return output

# Initializing the model
embed_size = 64  # Size of the embedding dimension
num_heads = 8    # Number of attention heads
model = AttentionModel(embed_size, num_heads)

# Generate input tensors for query, key, and value
batch_size = 1
seq_length = 10

query_tensor = torch.randn(batch_size, seq_length, embed_size)
key_tensor = torch.randn(batch_size, seq_length, embed_size)
value_tensor = torch.randn(batch_size, seq_length, embed_size)

# Forward pass through the model
output = model(query_tensor, key_tensor, value_tensor)

print("Output shape:", output.shape)
