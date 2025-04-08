import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, d_model, n_heads):
        super(AttentionModel, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value):
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale the dot product by the inverse scale
        inv_scale = self.d_model ** 0.5
        scaled_qk = qk / inv_scale
        
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        
        # Compute the dot product of the softmax output and the value tensor
        output = softmax_qk.matmul(value)
        return output

# Model Initialization
d_model = 64  # Dimension of the model
n_heads = 8   # Number of attention heads (not used in this simple model)
attention_model = AttentionModel(d_model, n_heads)

# Inputs to the model
batch_size = 1
seq_length = 10  # Length of the sequence
query = torch.randn(batch_size, seq_length, d_model)  # Query tensor
key = torch.randn(batch_size, seq_length, d_model)    # Key tensor
value = torch.randn(batch_size, seq_length, d_model)  # Value tensor

# Forward pass
output = attention_model(query, key, value)

# Output shape
print(output.shape)  # Should be (batch_size, seq_length, d_model)
