import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(AttentionModel, self).__init__()
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        self.num_heads = num_heads
        self.embed_size = embed_size
    
    def forward(self, query, key, value):
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(query, key.transpose(-2, -1))  # (batch_size, num_heads, seq_length, seq_length)
        
        # Scale the dot product by the inverse scale
        inv_scale = self.embed_size ** 0.5
        scaled_qk = qk / inv_scale
        
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        
        # Compute the dot product of the softmax output and the value tensor
        output = softmax_qk.matmul(value)  # (batch_size, num_heads, seq_length, embed_size)
        
        return output

# Example initialization
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads
model = AttentionModel(embed_size, num_heads)

# Inputs to the model
batch_size = 2
seq_length = 10
query = torch.randn(batch_size, num_heads, seq_length, embed_size)
key = torch.randn(batch_size, num_heads, seq_length, embed_size)
value = torch.randn(batch_size, num_heads, seq_length, embed_size)

# Forward pass
output = model(query, key, value)

print(output.shape)  # Should print the shape of the output tensor
