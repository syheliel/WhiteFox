import torch
import math

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(ScaledDotProductAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def forward(self, query, key, value):
        # Permute the dimensions of the query tensor
        q = query.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
        k = key.permute(0, 2, 1, 3)    # (batch_size, num_heads, seq_length, head_dim)
        v = value.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
        
        # Compute the dot product of the query and the transposed key, and divide by the square root of the last dimension of the query
        div = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))  # (batch_size, num_heads, seq_length, seq_length)
        
        # Convert the result to float32
        div = div.to(torch.float32)
        
        # Apply softmax to the result along the last dimension
        attn_weight = torch.softmax(div, dim=-1)
        
        # Convert the result to float16
        attn_weight = attn_weight.to(torch.float16)
        
        # Compute the dot product of the attention weights and the value
        output = attn_weight @ v  # (batch_size, num_heads, seq_length, head_dim)
        
        return output

# Initialize the model
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
model = ScaledDotProductAttention(embed_dim, num_heads)

# Inputs to the model
batch_size = 1
seq_length = 10  # Sequence length
head_dim = embed_dim // num_heads  # Dimension per head (assuming embed_dim is divisible by num_heads)

query = torch.randn(batch_size, seq_length, embed_dim)  # (batch_size, seq_length, embed_dim)
key = torch.randn(batch_size, seq_length, embed_dim)    # (batch_size, seq_length, embed_dim)
value = torch.randn(batch_size, seq_length, embed_dim)  # (batch_size, seq_length, embed_dim)

# Pass the inputs through the model
output = model(query, key, value)

print(output.shape)  # Output shape will be (batch_size, num_heads, seq_length, head_dim)
