import torch

# Model
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
        q = query.permute(0, 2, 1, 3)  # Permute the dimensions of the query tensor
        k = key.permute(0, 2, 1, 3)      # Permute the dimensions of the key tensor
        v = value.permute(0, 2, 1, 3)    # Permute the dimensions of the value tensor

        attention = torch.matmul(q, k.transpose(-2, -1))  # Dot product
        scaled_attention = attention.div(inv_scale)         # Scale the attention
        attention_weights = scaled_attention.softmax(dim=-1)  # Apply softmax
        output = attention_weights.matmul(v)                 # Weighted sum of value tensor
        return output

# Initialize the model
embed_size = 64  # Example embedding size
num_heads = 8    # Example number of heads
model = ScaledDotProductAttention(embed_size, num_heads)

# Inputs to the model
batch_size = 1
seq_length = 10
num_features = embed_size // num_heads  # Features per head

# Creating random input tensors for query, key, and value
query = torch.randn(batch_size, seq_length, num_heads, num_features)
key = torch.randn(batch_size, seq_length, num_heads, num_features)
value = torch.randn(batch_size, seq_length, num_heads, num_features)

# Setting the inverse scale
inv_scale = torch.tensor(embed_size ** -0.5)  # Example inverse scale

# Get the output from the model
output = model(query, key, value, inv_scale)

print(output.shape)  # Output shape
