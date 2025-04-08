import torch

# Description of requirements: Scaled Dot-Product Attention
class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.inv_scale = embed_size ** 0.5  # Inverse scale for attention

    def forward(self, query, key, value):
        q = query.permute(0, 2, 1, 3)  # Permute the dimensions of the query tensor
        k = key.permute(0, 2, 1, 3)     # Permute the dimensions of the key tensor
        v = value.permute(0, 2, 1, 3)   # Permute the dimensions of the value tensor
        
        attention = torch.matmul(q, k.transpose(-2, -1))  # Compute the dot product of the query and the transposed key
        scaled_attention = attention / self.inv_scale  # Scale the attention by dividing it by the inverse scale
        attention_weights = scaled_attention.softmax(dim=-1)  # Apply softmax to the scaled attention
        output = attention_weights.matmul(v)  # Compute the weighted sum of the value tensor
        
        return output

# Initializing the model with specific parameters
embed_size = 64  # Size of the embedding
num_heads = 8  # Number of attention heads
model = AttentionModel(embed_size, num_heads)

# Inputs to the model
batch_size = 2
seq_length = 10  # Length of the sequence
value_length = 10 # Length of the value sequence
query = torch.randn(batch_size, seq_length, num_heads, embed_size // num_heads)
key = torch.randn(batch_size, seq_length, num_heads, embed_size // num_heads)
value = torch.randn(batch_size, value_length, num_heads, embed_size // num_heads)

# Forward pass through the model
output = model(query, key, value)

# Output shape
print(output.shape)  # Should be (batch_size, seq_length, num_heads, embed_size // num_heads)
