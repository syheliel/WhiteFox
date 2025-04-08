import torch
import math

class SelfAttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.key_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.value_layer = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_layer(query).permute(0, 2, 1)  # Shape: (batch_size, seq_len, embed_dim)
        k = self.key_layer(key).permute(0, 2, 1)      # Shape: (batch_size, seq_len, embed_dim)
        v = self.value_layer(value).permute(0, 2, 1)  # Shape: (batch_size, seq_len, embed_dim)
        
        # Scale the query tensor by the square root of its last dimension size
        q = q / math.sqrt(q.size(-1))
        
        # Perform a matrix multiplication between the query tensor and the transposed key tensor
        div = q @ k.transpose(-2, -1)  # Shape: (batch_size, seq_len, seq_len)
        
        # Convert the result to float32
        div = div.to(torch.float32)
        
        # Apply softmax to the result along the last dimension
        attn_weight = torch.softmax(div, dim=-1)
        
        # Convert the result to float16
        attn_weight = attn_weight.to(torch.float16)
        
        # Perform a matrix multiplication between the attention weights and the value tensor
        output = attn_weight @ v  # Shape: (batch_size, seq_len, embed_dim)
        
        return output

# Initializing the model
embed_dim = 64  # Example embedding dimension
num_heads = 8   # Example number of attention heads
model = SelfAttentionModel(embed_dim=embed_dim, num_heads=num_heads)

# Inputs to the model (batch size of 1, sequence length of 10, embedding dimension of 64)
query = torch.randn(1, 10, embed_dim)
key = torch.randn(1, 10, embed_dim)
value = torch.randn(1, 10, embed_dim)

# Get the output from the model
output = model(query, key, value)
