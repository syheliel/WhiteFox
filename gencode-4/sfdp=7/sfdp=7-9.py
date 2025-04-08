import torch
import math

# Define the model class
class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.query_linear = torch.nn.Linear(embed_size, embed_size)
        self.key_linear = torch.nn.Linear(embed_size, embed_size)
        self.value_linear = torch.nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_linear(query).permute(0, 2, 1, 3)
        k = self.key_linear(key).permute(0, 2, 1, 3)
        v = self.value_linear(value).permute(0, 2, 1, 3)
        
        # Compute the dot product and scale
        div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        div = div.to(torch.float32)
        
        # Apply softmax to get attention weights
        attn_weight = torch.softmax(div, dim=-1)
        attn_weight = attn_weight.to(torch.float16)
        
        # Compute the output by applying attention weights to the value tensor
        output = attn_weight @ v
        return output

# Instantiate the model
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads
model = AttentionModel(embed_size, num_heads)

# Inputs to the model
batch_size = 1
seq_length = 10  # Length of the input sequences
value_length = 10  # Length of the value sequences

# Create random input tensors for query, key, and value
query = torch.randn(batch_size, seq_length, embed_size)  # (batch_size, seq_length, embed_size)
key = torch.randn(batch_size, seq_length, embed_size)    # (batch_size, seq_length, embed_size)
value = torch.randn(batch_size, value_length, embed_size) # (batch_size, value_length, embed_size)

# Get the output from the model
output = model(query, key, value)

# Print the output shape
print("Output shape:", output.shape)
