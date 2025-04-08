import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionModel(nn.Module):
    def __init__(self, embed_size, heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.dropout_p = dropout_p
        
        # Define linear transformations for query, key, and value
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, query, key, value):
        # Permute the dimensions of the query, key, and value tensors
        q = self.queries(query).permute(0, 2, 1, 3)  # (batch_size, heads, seq_len, embed_size // heads)
        k = self.keys(key).permute(0, 2, 1, 3)      # (batch_size, heads, seq_len, embed_size // heads)
        v = self.values(value).permute(0, 2, 1, 3)  # (batch_size, heads, seq_len, embed_size // heads)

        # Scale the query tensor by the square root of its last dimension size
        q = q / math.sqrt(q.size(-1))
        
        # Perform a matrix multiplication between the query tensor and the transposed key tensor
        div = q @ k.transpose(-2, -1)  # (batch_size, heads, seq_len, seq_len)

        # Convert the result to float32
        div = div.to(torch.float32)
        
        # Apply softmax to the result along the last dimension
        attn_weight = F.softmax(div, dim=-1)
        
        # Apply dropout to the softmax result
        attn_weight = self.dropout(attn_weight)
        
        # Convert the result to float16
        attn_weight = attn_weight.to(torch.float16)
        
        # Perform a matrix multiplication between the result and the value tensor
        out = attn_weight @ v  # (batch_size, heads, seq_len, embed_size // heads)
        
        return out

# Initialize the model
embed_size = 64  # Size of the embedding
heads = 8       # Number of attention heads
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_size, heads, dropout_p)

# Generate input tensors
batch_size = 2
seq_len = 10
query = torch.randn(batch_size, seq_len, embed_size)
key = torch.randn(batch_size, seq_len, embed_size)
value = torch.randn(batch_size, seq_len, embed_size)

# Forward pass through the model
output = model(query, key, value)

print(output.shape)  # Output tensor shape
