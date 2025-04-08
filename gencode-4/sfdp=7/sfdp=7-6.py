import torch
import math

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = embed_dim // num_heads
        self.query_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.key_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.value_layer = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_layer(query).permute(0, 2, 1, 3)
        k = self.key_layer(key).permute(0, 2, 1, 3)
        v = self.value_layer(value).permute(0, 2, 1, 3)

        # Compute the dot product of query and transposed key, and divide by sqrt of last dimension
        div = q @ k.transpose(-2, -1) / math.sqrt(self.depth)

        # Convert the result to float32
        div = div.to(torch.float32)

        # Apply softmax to the result along the last dimension
        attn_weight = torch.softmax(div, dim=-1)

        # Convert the result to float16
        attn_weight = attn_weight.to(torch.float16)

        # Compute the dot product of the attention weights and the value
        output = attn_weight @ v
        return output

# Initializing the model
embed_dim = 64  # Example embedding dimension
num_heads = 4   # Example number of attention heads
model = AttentionModel(embed_dim, num_heads)

# Inputs to the model
batch_size = 1
seq_length = 10  # Example sequence length
x_query = torch.randn(batch_size, seq_length, embed_dim)
x_key = torch.randn(batch_size, seq_length, embed_dim)
x_value = torch.randn(batch_size, seq_length, embed_dim)

# Running the model
output = model(x_query, x_key, x_value)

print("Output shape:", output.shape)
