import torch
import math

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        
        # Define linear layers for query, key, and value
        self.query_linear = torch.nn.Linear(embed_size, embed_size)
        self.key_linear = torch.nn.Linear(embed_size, embed_size)
        self.value_linear = torch.nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value):
        # Permute the input tensors
        q = self.query_linear(query).permute(0, 2, 1, 3)
        k = self.key_linear(key).permute(0, 2, 1, 3)
        v = self.value_linear(value).permute(0, 2, 1, 3)
        
        # Compute the dot product and scale
        div = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        div = div.to(torch.float32)  # Convert to float32
        
        # Apply softmax
        attn_weight = torch.softmax(div, dim=-1)
        attn_weight = attn_weight.to(torch.float16)  # Convert to float16
        
        # Compute the output
        output = attn_weight @ v
        return output

# Initialize the model with embedding size and number of heads
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads
model = AttentionModel(embed_size, num_heads)

# Inputs to the model
batch_size = 1  # Number of samples in a batch
seq_length = 10  # Length of the input sequence

# Create dummy input tensors for query, key, and value
query_tensor = torch.randn(batch_size, seq_length, embed_size)
key_tensor = torch.randn(batch_size, seq_length, embed_size)
value_tensor = torch.randn(batch_size, seq_length, embed_size)

# Get the output from the model
output = model(query_tensor, key_tensor, value_tensor)

print("Output shape:", output.shape)
