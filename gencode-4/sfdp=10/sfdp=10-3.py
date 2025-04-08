import torch

# Model definition
class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, query, key, value, inv_scale):
        q = query.permute(0, 2, 1, 3)  # Permute the dimensions of the query tensor
        k = key.permute(0, 2, 1, 3)      # Permute the dimensions of the key tensor
        v = value.permute(0, 2, 1, 3)    # Permute the dimensions of the value tensor
        
        attention = torch.matmul(q, k.transpose(-2, -1))  # Compute the dot product of the query and the transposed key
        scaled_attention = attention.div(inv_scale)         # Scale the attention by dividing it by the inverse scale
        attention_weights = scaled_attention.softmax(dim=-1) # Apply softmax to the scaled attention
        output = attention_weights.matmul(v)                 # Compute the weighted sum of the value tensor
        
        return output

# Initializing the model
embed_dim = 64  # Example embedding dimension
model = AttentionModel(embed_dim)

# Inputs to the model
batch_size = 1
seq_length = 10
num_heads = 4

# Generating random input tensors
query = torch.randn(batch_size, seq_length, num_heads, embed_dim)
key = torch.randn(batch_size, seq_length, num_heads, embed_dim)
value = torch.randn(batch_size, seq_length, num_heads, embed_dim)
inv_scale = torch.tensor(1.0 / (embed_dim ** 0.5))

# Forward pass through the model
output = model(query, key, value, inv_scale)

# Print output shape
print(output.shape)  # Should be (batch_size, seq_length, num_heads, embed_dim)
