import torch
import torch.nn as nn
import math

class AttentionModel(nn.Module):
    def __init__(self, input_dim, head_dim, dropout_p):
        super().__init__()
        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout_p)
        self.query_layer = nn.Linear(input_dim, head_dim)
        self.key_layer = nn.Linear(input_dim, head_dim)
        self.value_layer = nn.Linear(input_dim, head_dim)

    def forward(self, query, key, value):
        q = self.query_layer(query).permute(0, 2, 1)
        k = self.key_layer(key).permute(0, 2, 1)
        v = self.value_layer(value).permute(0, 2, 1)
        
        div = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        div = div.to(torch.float32)
        
        attn_weight = torch.softmax(div, dim=-1)
        attn_weight = self.dropout(attn_weight)
        attn_weight = attn_weight.to(torch.float16)
        
        output = attn_weight @ v
        return output

# Initialize the model with specific parameters
input_dim = 64  # Dimensionality of the input features
head_dim = 32   # Dimensionality of each attention head
dropout_p = 0.1 # Dropout probability

model = AttentionModel(input_dim, head_dim, dropout_p)

# Generate input tensors for query, key, and value
batch_size = 1
seq_length = 10

query = torch.randn(batch_size, seq_length, input_dim)
key = torch.randn(batch_size, seq_length, input_dim)
value = torch.randn(batch_size, seq_length, input_dim)

# Forward pass through the model
output = model(query, key, value)

# Print the output shape
print(output.shape)
