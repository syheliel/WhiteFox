import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super(SelfAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        # Define linear layers for query, key, and value transformations
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_layer(query).permute(0, 2, 1, 3)
        k = self.key_layer(key).permute(0, 2, 1, 3)
        v = self.value_layer(value).permute(0, 2, 1, 3)

        # Scale the query tensor by the square root of its last dimension size
        q = q / math.sqrt(q.size(-1))

        # Perform a matrix multiplication between the query tensor and the transposed key tensor
        div = q @ k.transpose(-2, -1)

        # Convert the result to float32
        div = div.to(torch.float32)

        # Apply softmax to the result along the last dimension
        attn_weight = F.softmax(div, dim=-1)

        # Apply dropout to the softmax result
        attn_weight = self.dropout(attn_weight)

        # Convert the result to float16
        attn_weight = attn_weight.to(torch.float16)

        # Perform a matrix multiplication between the result and the value tensor
        output = attn_weight @ v

        return output

# Initializing the model
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
dropout_p = 0.1 # Dropout probability
model = SelfAttentionModel(embed_dim, num_heads, dropout_p)

# Inputs to the model (batch_size, sequence_length, embed_dim)
query = torch.randn(1, 10, embed_dim)  # Query tensor
key = torch.randn(1, 10, embed_dim)    # Key tensor
value = torch.randn(1, 10, embed_dim)  # Value tensor

# Forward pass
output = model(query, key, value)

print(output.shape)  # Output shape
