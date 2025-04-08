import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(ScaledDotProductAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.inv_scale = torch.sqrt(torch.tensor(embed_size // num_heads, dtype=torch.float32))

    def forward(self, query, key, value):
        # Permute the dimensions of the query, key, and value tensors
        q = query.permute(0, 2, 1, 3)  # (batch_size, seq_length, num_heads, head_dim)
        k = key.permute(0, 2, 1, 3)    # (batch_size, seq_length, num_heads, head_dim)
        v = value.permute(0, 2, 1, 3)  # (batch_size, seq_length, num_heads, head_dim)

        # Compute the dot product of the query and the transposed key
        attention = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, num_heads, seq_length, seq_length)

        # Scale the attention by dividing it by the inverse scale
        scaled_attention = attention / self.inv_scale

        # Apply softmax to the scaled attention to obtain attention weights
        attention_weights = scaled_attention.softmax(dim=-1)  # (batch_size, num_heads, seq_length, seq_length)

        # Compute the weighted sum of the value tensor
        output = attention_weights.matmul(v)  # (batch_size, num_heads, seq_length, head_dim)

        return output

# Initializing the model
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads
model = ScaledDotProductAttention(embed_size, num_heads)

# Inputs to the model
batch_size = 1
seq_length = 10
query = torch.randn(batch_size, seq_length, embed_size)  # (batch_size, seq_length, embed_size)
key = torch.randn(batch_size, seq_length, embed_size)    # (batch_size, seq_length, embed_size)
value = torch.randn(batch_size, seq_length, embed_size)  # (batch_size, seq_length, embed_size)

# Forward pass through the model
output = model(query, key, value)

# Display the output shape
print(output.shape)
