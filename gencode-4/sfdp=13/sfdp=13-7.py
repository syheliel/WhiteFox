import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.key_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.value_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        # Permute dimensions of query, key, and value tensors
        q = self.query_proj(query).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len_q, head_dim)
        k = self.key_proj(key).permute(0, 2, 1, 3)      # (batch_size, num_heads, seq_len_k, head_dim)
        v = self.value_proj(value).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len_v, head_dim)

        # Calculate attention scores
        t1 = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, num_heads, seq_len_q, seq_len_k)
        inv_scale = self.embed_dim ** 0.5
        t2 = t1 / inv_scale  # Scale the scores
        if attn_mask is not None:
            t3 = t2 + attn_mask  # Add the attention mask
        else:
            t3 = t2
        t4 = t3.softmax(dim=-1)  # Apply softmax to the scores
        t5 = t4.matmul(v)  # Matrix multiplication with the value tensor
        return t5 

# Initialize the model
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
model = AttentionModel(embed_dim, num_heads)

# Inputs to the model
batch_size = 2
seq_len = 10
query = torch.randn(batch_size, seq_len, embed_dim)  # (batch_size, seq_len_q, embed_dim)
key = torch.randn(batch_size, seq_len, embed_dim)    # (batch_size, seq_len_k, embed_dim)
value = torch.randn(batch_size, seq_len, embed_dim)  # (batch_size, seq_len_v, embed_dim)
attn_mask = torch.randn(batch_size, num_heads, seq_len, seq_len)  # Attention mask

# Forward pass
output = model(query, key, value, attn_mask)
print(output.shape)  # Expected output shape: (batch_size, num_heads, seq_len_q, head_dim)
