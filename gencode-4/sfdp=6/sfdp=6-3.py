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
        # Permute the dimensions of the query, key, and value tensors
        q = query.permute(0, 2, 1, 3)  # (batch_size, embed_size, seq_length, heads)
        k = key.permute(0, 2, 1, 3)    # (batch_size, embed_size, seq_length, heads)
        v = value.permute(0, 2, 1, 3)  # (batch_size, embed_size, seq_length, heads)

        # Compute the dot product of the query and the transposed key, divided by the square root of the last dimension
        div = (q @ k.transpose(-2, -1)) / math.sqrt(self.embed_size)
        div = div.to(torch.float32)  # Convert the tensor to float32

        # Apply softmax to the last dimension of the tensor
        attn_weight = F.softmax(div, dim=-1)
        
        # Apply dropout to the tensor
        attn_weight = F.dropout(attn_weight, p=self.dropout_p, training=self.training)
        
        # Convert the tensor to float16
        attn_weight = attn_weight.to(torch.float16)

        # Compute the dot product of the attention weights and the value
        output = attn_weight @ v
        return output

# Initialize the model parameters
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads
dropout_p = 0.1  # Dropout probability

# Initialize the model
model = AttentionModel(embed_size, num_heads, dropout_p)

# Create input tensors
batch_size = 1
seq_length = 10  # Length of the input sequences

query = torch.randn(batch_size, embed_size, seq_length, num_heads)
key = torch.randn(batch_size, embed_size, seq_length, num_heads)
value = torch.randn(batch_size, embed_size, seq_length, num_heads)

# Run the model with the inputs
output = model(query, key, value)

# Display the output shape
print(output.shape)  # Expected shape: (batch_size, embed_size, seq_length, heads)
