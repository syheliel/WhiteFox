import torch
import torch.nn.functional as F
import math

# Define the model
class SelfAttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.dropout_p = 0.1  # Dropout probability

        # Linear transformations for query, key, and value
        self.query_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.key_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.value_linear = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_linear(query).permute(0, 2, 1, 3)  # Shape: (batch_size, heads, seq_len, head_dim)
        k = self.key_linear(key).permute(0, 2, 1, 3)
        v = self.value_linear(value).permute(0, 2, 1, 3)

        # Scale the query tensor
        q = q / math.sqrt(q.size(-1))

        # Matrix multiplication between the query and transposed key
        div = q @ k.transpose(-2, -1)  # Shape: (batch_size, heads, seq_len, seq_len)

        # Convert result to float32
        div = div.to(torch.float32)

        # Apply softmax to the result along the last dimension
        attn_weight = F.softmax(div, dim=-1)

        # Apply dropout to the softmax result
        attn_weight = F.dropout(attn_weight, p=self.dropout_p, training=self.training)

        # Convert the result to float16
        attn_weight = attn_weight.to(torch.float16)

        # Perform matrix multiplication with the value tensor
        output = attn_weight @ v  # Shape: (batch_size, heads, seq_len, head_dim)

        return output

# Initialize the model
embed_dim = 64  # Embedding dimension
heads = 4  # Number of attention heads
model = SelfAttentionModel(embed_dim, heads)

# Generate input tensors for query, key, and value
batch_size = 2
seq_len = 10
input_tensor = torch.randn(batch_size, seq_len, embed_dim)

# Forward pass through the model
output = model(input_tensor, input_tensor, input_tensor)

# Print the output shape
print("Output shape:", output.shape)  # Should be (batch_size, heads, seq_len, head_dim)
