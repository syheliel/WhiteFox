import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = embed_dim ** 0.5  # Compute the scale for attention
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, attn_mask):
        q = self.query(query).permute(0, 2, 1, 3)  # Shape: (batch_size, num_heads, seq_length, head_dim)
        k = self.key(key).permute(0, 2, 1, 3)      # Shape: (batch_size, num_heads, seq_length, head_dim)
        v = self.value(value).permute(0, 2, 1, 3)  # Shape: (batch_size, num_heads, seq_length, head_dim)

        t1 = torch.matmul(q, k.transpose(-2, -1))  # Shape: (batch_size, num_heads, seq_length, seq_length)
        t2 = t1 / self.scale                         # Apply the inverse scale
        t3 = t2 + attn_mask                         # Add the attention mask
        t4 = t3.softmax(dim=-1)                     # Apply softmax
        t5 = t4.matmul(v)                           # Shape: (batch_size, num_heads, seq_length, head_dim)
        
        return t5

# Initialize the model
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
model = AttentionModel(embed_dim, num_heads)

# Generate a random input tensor for query, key, and value
batch_size = 2
seq_length = 10
attn_mask = torch.randn(batch_size, seq_length, seq_length)  # Attention mask

query = torch.randn(batch_size, seq_length, embed_dim)  # Query tensor
key = torch.randn(batch_size, seq_length, embed_dim)    # Key tensor
value = torch.randn(batch_size, seq_length, embed_dim)  # Value tensor

# Forward pass through the model
output = model(query, key, value, attn_mask)

# Print the output shape
print(output.shape)  # Expected output shape: (batch_size, num_heads, seq_length, head_dim)
