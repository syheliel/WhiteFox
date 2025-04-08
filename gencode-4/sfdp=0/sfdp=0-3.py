import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.query_layer = nn.Linear(d_model, d_model)
        self.key_layer = nn.Linear(d_model, d_model)
        self.value_layer = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_model = d_model

    def forward(self, query, key, value):
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(self.query_layer(query), self.key_layer(key).transpose(-2, -1))
        
        # Scale the dot product by the inverse scale
        inv_scale = self.d_model ** 0.5
        scaled_qk = qk / inv_scale
        
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        
        # Compute the dot product of the softmax output and the value tensor
        output = softmax_qk.matmul(self.value_layer(value))
        return output

# Initializing the model
d_model = 64  # Dimensionality of the input features
n_heads = 8   # Number of attention heads
model = AttentionModel(d_model, n_heads)

# Inputs to the model
batch_size = 2
seq_length = 10

# Creating random input tensors for query, key, and value
query = torch.randn(batch_size, seq_length, d_model)
key = torch.randn(batch_size, seq_length, d_model)
value = torch.randn(batch_size, seq_length, d_model)

# Getting the output from the model
output = model(query, key, value)

# Output shape
print(output.shape)  # Should be (batch_size, seq_length, d_model)
