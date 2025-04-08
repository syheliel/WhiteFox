import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.query_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.key_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.value_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.scale = embed_dim ** 0.5  # Scaling factor for the dot product

    def forward(self, query, key, value, attn_mask):
        q = self.query_layer(query).view(query.size(0), -1, self.num_heads, query.size(-1)).permute(0, 2, 1, 3)
        k = self.key_layer(key).view(key.size(0), -1, self.num_heads, key.size(-1)).permute(0, 2, 1, 3)
        v = self.value_layer(value).view(value.size(0), -1, self.num_heads, value.size(-1)).permute(0, 2, 1, 3)
        
        t1 = torch.matmul(q, k.transpose(-2, -1))
        t2 = t1 / self.scale
        t3 = t2 + attn_mask
        t4 = t3.softmax(dim=-1)
        t5 = t4.matmul(v)
        
        return t5

# Example of initializing the model
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
model = AttentionModel(embed_dim, num_heads)

# Inputs to the model
batch_size = 1
seq_length = 10
# Create random input tensors
query = torch.randn(batch_size, seq_length, embed_dim)
key = torch.randn(batch_size, seq_length, embed_dim)
value = torch.randn(batch_size, seq_length, embed_dim)
attn_mask = torch.zeros(batch_size, num_heads, seq_length, seq_length)  # Attention mask (can be adjusted as needed)

# Forward pass
output = model(query, key, value, attn_mask)

# Print the output shape
print(output.shape)  # Should be (batch_size, num_heads, seq_length, embed_dim)
