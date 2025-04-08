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

    def forward(self, query, key, value):
        # Permute the dimensions of the query tensor
        q = query.permute(0, 2, 1, 3)  # (batch_size, seq_length, heads, embed_size // heads)
        k = key.permute(0, 2, 1, 3)    # (batch_size, seq_length, heads, embed_size // heads)
        v = value.permute(0, 2, 1, 3)  # (batch_size, seq_length, heads, embed_size // heads)

        # Scale the query tensor by the square root of its last dimension size
        q = q / math.sqrt(q.size(-1))

        # Perform a matrix multiplication between the query tensor and the transposed key tensor
        div = q @ k.transpose(-2, -1)  # (batch_size, heads, seq_length, seq_length)

        # Convert the result to float32
        div = div.to(torch.float32)

        # Apply softmax to the result along the last dimension
        attn_weight = F.softmax(div, dim=-1)

        # Apply dropout to the softmax result
        attn_weight = F.dropout(attn_weight, p=self.dropout_p, training=self.training)

        # Convert the result to float16
        attn_weight = attn_weight.to(torch.float16)

        # Perform a matrix multiplication between the result and the value tensor
        output = attn_weight @ v  # (batch_size, heads, seq_length, embed_size // heads)

        return output

# Initializing the model
embed_size = 64  # Embedding size
heads = 8        # Number of attention heads
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_size, heads, dropout_p)

# Inputs to the model
batch_size = 1
seq_length = 10
num_features = embed_size  # Number of features (embedding size)

# Create random input tensors for query, key, and value
query = torch.randn(batch_size, seq_length, num_features).to(torch.float32)
key = torch.randn(batch_size, seq_length, num_features).to(torch.float32)
value = torch.randn(batch_size, seq_length, num_features).to(torch.float32)

# Getting the output from the model
output = model(query, key, value)

# Print the shape of the output
print(output.shape)  # Expected shape: (batch_size, heads, seq_length, embed_size // heads)
