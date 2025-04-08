import torch

class SelfAttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, inv_scale=1.0):
        # Permute the dimensions of the query, key, and value tensors
        q = query.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)
        k = key.permute(0, 2, 1, 3)    # (batch_size, seq_len, num_heads, head_dim)
        v = value.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)

        # Perform matrix multiplication and apply scaling
        t1 = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)
        t2 = t1.div(inv_scale)                      # Divide by inverse scale
        
        # Add attention mask if provided
        if attn_mask is not None:
            t3 = t2 + attn_mask                     # Add attention mask
        else:
            t3 = t2

        # Apply softmax
        t4 = t3.softmax(dim=-1)                    # (batch_size, num_heads, seq_len, seq_len)

        # Matrix multiplication with value tensor
        t5 = t4.matmul(v)                          # (batch_size, num_heads, seq_len, head_dim)
        
        return t5


# Model initialization
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
model = SelfAttentionModel(embed_dim, num_heads)

# Create input tensors for query, key, and value
batch_size = 1
seq_len = 10
head_dim = embed_dim // num_heads

query_tensor = torch.randn(batch_size, seq_len, num_heads, head_dim)
key_tensor = torch.randn(batch_size, seq_len, num_heads, head_dim)
value_tensor = torch.randn(batch_size, seq_len, num_heads, head_dim)

# Optional: attention mask (batch_size, 1, 1, seq_len)
attn_mask = torch.zeros(batch_size, 1, 1, seq_len)  # Example of no masking

# Forward pass
output = model(query_tensor, key_tensor, value_tensor, attn_mask)

print("Output shape:", output.shape)  # Should be (batch_size, num_heads, seq_len, head_dim)
