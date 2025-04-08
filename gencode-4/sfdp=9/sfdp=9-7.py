import torch
import math

# Define the model
class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Define linear layers for query, key, and value
        self.query_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.key_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.value_linear = torch.nn.Linear(embed_dim, embed_dim)

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

        # Convert the result to float16
        attn_weight = attn_weight.to(torch.float16)

        # Perform matrix multiplication between the attention weights and the value tensor
        return attn_weight @ v

# Initialization of the model
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
model = AttentionModel(embed_dim, num_heads)

# Generate inputs for the model
batch_size = 1
seq_len = 10  # Sequence length
x_query = torch.randn(batch_size, seq_len, embed_dim)  # Query input
x_key = torch.randn(batch_size, seq_len, embed_dim)    # Key input
x_value = torch.randn(batch_size, seq_len, embed_dim)  # Value input

# Forward pass
output = model(x_query, x_key, x_value)

# Output result
print(output.shape)  # Should be (batch_size, num_heads, seq_len, embed_dim // num_heads)
