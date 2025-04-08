import torch

# Define the model
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size must be divisible by num_heads"

    def forward(self, query, key, value, inv_scale):
        # Permute the dimensions of the query, key, and value tensors
        q = query.permute(0, 2, 1, 3)  # Shape: (batch_size, num_heads, seq_length, head_dim)
        k = key.permute(0, 2, 1, 3)    # Shape: (batch_size, num_heads, seq_length, head_dim)
        v = value.permute(0, 2, 1, 3)  # Shape: (batch_size, num_heads, seq_length, head_dim)

        # Compute the dot product of the query and the transposed key
        attention = torch.matmul(q, k.transpose(-2, -1))  # Shape: (batch_size, num_heads, seq_length, seq_length)

        # Scale the attention by dividing it by the inverse scale
        scaled_attention = attention.div(inv_scale)

        # Apply softmax to the scaled attention
        attention_weights = scaled_attention.softmax(dim=-1)  # Shape: (batch_size, num_heads, seq_length, seq_length)

        # Compute the weighted sum of the value tensor
        output = attention_weights.matmul(v)  # Shape: (batch_size, num_heads, seq_length, head_dim)

        return output

# Example usage:
# Initialize the model
embed_size = 64  # Size of embedding
num_heads = 8    # Number of attention heads
inv_scale = embed_size ** 0.5  # Inverse scale for attention

model = ScaledDotProductAttention(embed_size, num_heads)

# Generate example input tensors
batch_size = 1
seq_length = 10  # Length of the input sequence
query = torch.randn(batch_size, seq_length, embed_size)  # Shape: (1, 10, 64)
key = torch.randn(batch_size, seq_length, embed_size)    # Shape: (1, 10, 64)
value = torch.randn(batch_size, seq_length, embed_size)  # Shape: (1, 10, 64)

# Forward pass through the model
output = model(query, key, value, inv_scale)

# Check the output shape
print(output.shape)  # Expected shape: (1, 8, 10, 8)
