import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(AttentionModel, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value):
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale the dot product by the inverse scale (sqrt of the dimension of key)
        inv_scale = key.size(-1) ** 0.5
        scaled_qk = qk / inv_scale
        
        # Apply softmax to the scaled dot product
        softmax_qk = F.softmax(scaled_qk, dim=-1)
        
        # Compute the dot product of the softmax output and the value tensor
        output = softmax_qk.matmul(value)
        return output

# Initialize the model with an embedding size of 64 and 8 heads
embed_size = 64
num_heads = 8
model = AttentionModel(embed_size, num_heads)

# Generating input tensors
batch_size = 1
seq_length = 10  # Example sequence length
query = torch.randn(batch_size, seq_length, embed_size)
key = torch.randn(batch_size, seq_length, embed_size)
value = torch.randn(batch_size, seq_length, embed_size)

# Forward pass through the model
output = model(query, key, value)

# Check the output shape
print(output.shape)  # Expected shape: (batch_size, seq_length, embed_size)
