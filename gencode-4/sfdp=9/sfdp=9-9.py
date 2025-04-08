import torch
import math

# Define the model
class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, num_heads):
        super(AttentionModel, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads

    def forward(self, query, key, value):
        # Permute the dimensions of the query, key, and value tensors
        q = query.permute(0, 2, 1, 3)  # (batch_size, seq_length, num_heads, embed_size // num_heads)
        k = key.permute(0, 2, 1, 3)    # (batch_size, seq_length, num_heads, embed_size // num_heads)
        v = value.permute(0, 2, 1, 3)  # (batch_size, seq_length, num_heads, embed_size // num_heads)

        # Scale the query tensor by the square root of its last dimension size
        q = q / math.sqrt(q.size(-1))

        # Perform matrix multiplication between the query tensor and the transposed key tensor
        div = q @ k.transpose(-2, -1)  # (batch_size, num_heads, seq_length, seq_length)

        # Convert the result to float32
        div = div.to(torch.float32)

        # Apply softmax to the result along the last dimension
        attn_weight = torch.softmax(div, dim=-1)

        # Convert the result to float16
        attn_weight = attn_weight.to(torch.float16)

        # Perform matrix multiplication between the attention weights and the value tensor
        output = attn_weight @ v  # (batch_size, num_heads, seq_length, embed_size // num_heads)

        return output

# Initialize the model
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads
model = AttentionModel(embed_size, num_heads)

# Generate example input tensors
batch_size = 1
seq_length = 10  # Length of the sequence
query = torch.randn(batch_size, seq_length, num_heads, embed_size // num_heads)
key = torch.randn(batch_size, seq_length, num_heads, embed_size // num_heads)
value = torch.randn(batch_size, seq_length, num_heads, embed_size // num_heads)

# Forward pass
output = model(query, key, value)

# Display the output shape
print(output.shape)  # Expected output shape: (1, num_heads, seq_length, embed_size // num_heads)
