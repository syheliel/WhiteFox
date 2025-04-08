import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, d_model, num_heads):
        super(AttentionModel, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale the dot product by the inverse scale
        inv_scale = key.size(-1) ** 0.5
        scaled_qk = qk / inv_scale
        
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        
        # Compute the dot product of the softmax output and the value tensor
        output = softmax_qk.matmul(value)
        
        return output

# Initialize the model
d_model = 64  # Dimension of the model
num_heads = 8  # Number of attention heads
model = AttentionModel(d_model, num_heads)

# Generate random input tensors for query, key, and value
batch_size = 2
seq_length = 10  # Sequence length
query = torch.randn(batch_size, seq_length, d_model)
key = torch.randn(batch_size, seq_length, d_model)
value = torch.randn(batch_size, seq_length, d_model)

# Forward pass through the model
output = model(query, key, value)

print("Output shape:", output.shape)  # Should be (batch_size, seq_length, d_model)
