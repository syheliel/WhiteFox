import torch
import math

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.key_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.value_linear = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_linear(query).permute(0, 2, 1)
        k = self.key_linear(key).permute(0, 2, 1)
        v = self.value_linear(value).permute(0, 2, 1)

        # Scale the query tensor by the square root of its last dimension size
        q = q / math.sqrt(q.size(-1))

        # Perform matrix multiplication between the query and transposed key tensors
        div = q @ k.transpose(-2, -1)

        # Convert the result to float32
        div = div.to(torch.float32)

        # Apply softmax to the result along the last dimension
        attn_weight = torch.softmax(div, dim=-1)

        # Convert the result to float16
        attn_weight = attn_weight.to(torch.float16)

        # Perform matrix multiplication between the attention weights and the value tensor
        output = attn_weight @ v
        return output

# Initializing the model
embed_dim = 64
num_heads = 8
attention_model = AttentionModel(embed_dim, num_heads)

# Inputs to the model
batch_size = 1
seq_length = 10
input_tensor = torch.randn(batch_size, seq_length, embed_dim)

# Using the same tensor for query, key, and value for this example
output = attention_model(input_tensor, input_tensor, input_tensor)

print(output)
