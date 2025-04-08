import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.inv_scale_factor = embed_dim ** 0.5  # Inverse scale factor

    def forward(self, query, key, value):
        qk = torch.matmul(self.query_linear(query), self.key_linear(key).transpose(-2, -1))  # Compute the dot product
        scaled_qk = qk / self.inv_scale_factor  # Scale the dot product
        softmax_qk = F.softmax(scaled_qk, dim=-1)  # Apply softmax
        dropout_qk = self.dropout(softmax_qk)  # Apply dropout
        output = torch.matmul(dropout_qk, self.value_linear(value))  # Compute the final output
        return output

# Initializing the model
embed_dim = 64
num_heads = 8
dropout_p = 0.1
model = AttentionModel(embed_dim, num_heads, dropout_p)

# Input tensors
batch_size = 1
seq_length = 10  # Number of tokens in the sequence
query = torch.randn(batch_size, seq_length, embed_dim)  # Query tensor
key = torch.randn(batch_size, seq_length, embed_dim)    # Key tensor
value = torch.randn(batch_size, seq_length, embed_dim)  # Value tensor

# Forward pass
output = model(query, key, value)
print(output)
