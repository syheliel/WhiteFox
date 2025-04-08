import torch
import torch.nn as nn
import math

# Define the model class
class SelfAttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super(SelfAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # Apply linear transformations to create query, key, and value
        q = self.query(x).view(x.size(0), -1, self.num_heads, self.embed_dim // self.num_heads)
        k = self.key(x).view(x.size(0), -1, self.num_heads, self.embed_dim // self.num_heads)
        v = self.value(x).view(x.size(0), -1, self.num_heads, self.embed_dim // self.num_heads)

        # Permute the dimensions of the query, key, and value tensors
        q = q.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
        k = k.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
        v = v.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)

        # Compute the dot product of the query and the transposed key
        div = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))  # (batch_size, num_heads, seq_length, seq_length)

        # Convert the tensor to float32
        div = div.to(torch.float32)

        # Apply softmax to the last dimension
        attn_weight = torch.softmax(div, dim=-1)  # (batch_size, num_heads, seq_length, seq_length)

        # Apply dropout to the attention weights
        attn_weight = self.dropout(attn_weight)

        # Convert the tensor to float16
        attn_weight = attn_weight.to(torch.float16)

        # Compute the dot product of the attention weights and the value
        output = attn_weight @ v  # (batch_size, num_heads, seq_length, head_dim)

        return output

# Initialize model parameters
embed_dim = 64
num_heads = 8
dropout_p = 0.1

# Create an instance of the model
model = SelfAttentionModel(embed_dim, num_heads, dropout_p)

# Generate an input tensor
batch_size = 1
seq_length = 10
input_tensor = torch.randn(batch_size, seq_length, embed_dim)

# Get the output from the model
output = model(input_tensor)

print("Output shape:", output.shape)  # Should be (batch_size, num_heads, seq_length, head_dim)
