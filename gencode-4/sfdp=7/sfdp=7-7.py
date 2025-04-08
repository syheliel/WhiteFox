import torch
import math

class AttentionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_layer = torch.nn.Linear(64, 64)  # Linear layer for query
        self.key_layer = torch.nn.Linear(64, 64)    # Linear layer for key
        self.value_layer = torch.nn.Linear(64, 64)  # Linear layer for value

    def forward(self, query, key, value):
        q = self.query_layer(query).permute(0, 2, 1, 3)  # Shape: (batch_size, seq_len, num_heads, head_dim)
        k = self.key_layer(key).permute(0, 2, 1, 3)      # Shape: (batch_size, seq_len, num_heads, head_dim)
        v = self.value_layer(value).permute(0, 2, 1, 3)  # Shape: (batch_size, seq_len, num_heads, head_dim)
        
        div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))  # Dot product and scaling
        div = div.to(torch.float32)                               # Convert to float32
        attn_weight = torch.softmax(div, dim=-1)                # Softmax on last dimension
        attn_weight = attn_weight.to(torch.float16)             # Convert to float16
        output = attn_weight @ v                                 # Dot product with value
        
        return output

# Initializing the model
model = AttentionModel()

# Inputs to the model
batch_size = 1
seq_len = 10
num_heads = 2
head_dim = 32

# Create input tensors with the appropriate shapes
query_input = torch.randn(batch_size, seq_len, num_heads, head_dim)
key_input = torch.randn(batch_size, seq_len, num_heads, head_dim)
value_input = torch.randn(batch_size, seq_len, num_heads, head_dim)

# Forward pass through the model
output = model(query_input, key_input, value_input)

# Output shape
print(output.shape)  # Should be (batch_size, num_heads, seq_len, head_dim)
