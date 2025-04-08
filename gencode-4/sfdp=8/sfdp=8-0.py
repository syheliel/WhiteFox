import torch
import torch.nn as nn
import math

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        
        # Define linear layers for query, key, and value
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)
        
        # Define dropout layer
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_linear(query).permute(0, 2, 1, 3)
        k = self.key_linear(key).permute(0, 2, 1, 3)
        v = self.value_linear(value).permute(0, 2, 1, 3)

        # Scale the query tensor by the square root of its last dimension size
        q = q / math.sqrt(q.size(-1))

        # Perform matrix multiplication between the query tensor and the transposed key tensor
        div = q @ k.transpose(-2, -1)

        # Convert the result to float32
        div = div.to(torch.float32)

        # Apply softmax to the result along the last dimension
        attn_weight = torch.softmax(div, dim=-1)

        # Apply dropout to the softmax result
        attn_weight = self.dropout(attn_weight)

        # Convert the result to float16
        attn_weight = attn_weight.to(torch.float16)

        # Perform matrix multiplication between the result and the value tensor
        output = attn_weight @ v
        
        return output

# Initializing the model
embed_dim = 64    # Size of each embedding
num_heads = 8     # Number of attention heads
dropout_p = 0.1   # Dropout probability
model = AttentionModel(embed_dim, num_heads, dropout_p)

# Inputs to the model: batch size of 1, sequence length of 10, and embedding size of 64
query = torch.randn(1, 10, embed_dim)
key = torch.randn(1, 10, embed_dim)
value = torch.randn(1, 10, embed_dim)

# Forward pass through the model
output = model(query, key, value)

# Output tensor shape
print(output.shape)  # Should be [1, 10, num_heads, embed_dim/num_heads]
