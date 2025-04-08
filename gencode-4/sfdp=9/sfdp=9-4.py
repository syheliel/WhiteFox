import torch
import math

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads

    def forward(self, query, key, value):
        # Permute the dimensions of the query tensor
        q = query.permute(0, 2, 1, 3)
        # Permute the dimensions of the key tensor
        k = key.permute(0, 2, 1, 3)
        # Permute the dimensions of the value tensor
        v = value.permute(0, 2, 1, 3)

        # Scale the query tensor by the square root of its last dimension size
        q = q / math.sqrt(q.size(-1))

        # Perform a matrix multiplication between the query tensor and the transposed key tensor
        div = q @ k.transpose(-2, -1)

        # Convert the result to float32
        div = div.to(torch.float32)

        # Apply softmax to the result along the last dimension
        attn_weight = torch.softmax(div, dim=-1)

        # Convert the result to float16
        attn_weight = attn_weight.to(torch.float16)

        # Perform a matrix multiplication between the attention weights and the value tensor
        output = attn_weight @ v

        return output

# Initializing the model
embed_size = 64  # Example embedding size
num_heads = 8    # Example number of attention heads
model = AttentionModel(embed_size, num_heads)

# Inputs to the model
batch_size = 1
seq_length = 10  # Sequence length
num_features = embed_size // num_heads  # Features per head

query = torch.randn(batch_size, seq_length, embed_size, num_heads)  # Shape: (batch_size, seq_length, embed_size, num_heads)
key = torch.randn(batch_size, seq_length, embed_size, num_heads)    # Same shape as query
value = torch.randn(batch_size, seq_length, embed_size, num_heads)  # Same shape as query

# Forward pass through the model
output = model(query, key, value)

# Print the output shape
print(output.shape)  # Should print: (batch_size, seq_length, num_heads, features_per_head)
