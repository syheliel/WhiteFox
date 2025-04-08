import torch
import torch.nn as nn
import math

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(AttentionModel, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value):
        q = self.query_linear(query).permute(0, 2, 1, 3)  # Permute the dimensions of the query tensor
        k = self.key_linear(key).permute(0, 2, 1, 3)      # Permute the dimensions of the key tensor
        v = self.value_linear(value).permute(0, 2, 1, 3)  # Permute the dimensions of the value tensor
        
        div = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))  # Compute scaled dot product
        div = div.to(torch.float32)  # Convert to float32
        attn_weight = torch.softmax(div, dim=-1)  # Apply softmax
        attn_weight = attn_weight.to(torch.float16)  # Convert to float16
        
        output = attn_weight @ v  # Compute the dot product of attention weights and value
        return output

# Initialize the model
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads
model = AttentionModel(embed_size, num_heads)

# Generate input tensors
batch_size = 1
seq_length = 10  # Length of the sequence
input_tensor = torch.randn(batch_size, seq_length, embed_size)

# Pass the same input tensor for query, key, and value
output = model(input_tensor, input_tensor, input_tensor)
print(output.shape)  # Output shape
