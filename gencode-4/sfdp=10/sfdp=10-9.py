import torch

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size

    def forward(self, query, key, value, inv_scale):
        # Permute the dimensions of the query, key, and value tensors
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)

        # Compute the dot product of the query and the transposed key
        attention = torch.matmul(q, k.transpose(-2, -1))

        # Scale the attention by dividing it by the inverse scale
        scaled_attention = attention.div(inv_scale)

        # Apply softmax to the scaled attention
        attention_weights = scaled_attention.softmax(dim=-1)

        # Compute the weighted sum of the value tensor
        output = attention_weights.matmul(v)
        return output

# Initializing the model with an embedding size (for example, 64)
embed_size = 64
model = ScaledDotProductAttention(embed_size)

# Inputs to the model
# Creating random input tensors for query, key, and value
batch_size = 1
seq_length = 10
num_heads = 4  # Number of attention heads
inv_scale = 1 / (embed_size ** 0.5)  # Inverse scale for attention

query = torch.randn(batch_size, seq_length, num_heads, embed_size)
key = torch.randn(batch_size, seq_length, num_heads, embed_size)
value = torch.randn(batch_size, seq_length, num_heads, embed_size)

# Forward pass through the model
output = model(query, key, value, inv_scale)
