import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionModel(nn.Module):
    def __init__(self, dropout_p=0.1):
        super().__init__()
        self.dropout_p = dropout_p
        self.query_layer = nn.Linear(64, 64)  # Assuming input feature size is 64
        self.key_layer = nn.Linear(64, 64)
        self.value_layer = nn.Linear(64, 64)

    def forward(self, query, key, value):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_layer(query).permute(0, 2, 1)  # Shape: (batch_size, seq_len, feature_dim)
        k = self.key_layer(key).permute(0, 2, 1)      # Shape: (batch_size, seq_len, feature_dim)
        v = self.value_layer(value).permute(0, 2, 1)  # Shape: (batch_size, seq_len, feature_dim)

        # Compute the dot product of the query and the transposed key, and divide by sqrt of last dimension
        div = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))  # Shape: (batch_size, seq_len, seq_len)

        # Convert the tensor to float32
        div = div.to(torch.float32)

        # Apply softmax to the last dimension of the tensor
        attn_weight = F.softmax(div, dim=-1)  # Shape: (batch_size, seq_len, seq_len)

        # Apply dropout to the tensor
        attn_weight = F.dropout(attn_weight, p=self.dropout_p, training=self.training)

        # Convert the tensor to float16
        attn_weight = attn_weight.to(torch.float16)

        # Compute the dot product of the attention weights and the value
        output = attn_weight @ v  # Shape: (batch_size, seq_len, feature_dim)
        return output

# Initializing the model
model = AttentionModel(dropout_p=0.1)

# Creating input tensors
# Assuming input feature size is 64 and sequence length is 10
batch_size = 2
seq_len = 10
feature_dim = 64

query_input = torch.randn(batch_size, seq_len, feature_dim)
key_input = torch.randn(batch_size, seq_len, feature_dim)
value_input = torch.randn(batch_size, seq_len, feature_dim)

# Forward pass through the model
output = model(query_input, key_input, value_input)

# Displaying the shape of the output
print(output.shape)  # Expected shape: (batch_size, seq_len, feature_dim)
