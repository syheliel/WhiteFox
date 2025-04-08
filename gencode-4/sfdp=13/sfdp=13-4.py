import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionModel, self).__init__()
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.scale = embed_dim ** 0.5  # Scaling factor for the attention scores

    def forward(self, query, key, value, attn_mask):
        # Permute the dimensions for batch processing
        q = self.query_linear(query).permute(0, 2, 1, 3)  # Shape: (batch_size, num_heads, seq_len_q, head_dim)
        k = self.key_linear(key).permute(0, 2, 1, 3)      # Shape: (batch_size, num_heads, seq_len_k, head_dim)
        v = self.value_linear(value).permute(0, 2, 1, 3)  # Shape: (batch_size, num_heads, seq_len_v, head_dim)

        # Compute attention scores
        t1 = torch.matmul(q, k.transpose(-2, -1))  # Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        t2 = t1 / self.scale                          # Scale the attention scores
        t3 = t2 + attn_mask                          # Apply attention mask
        t4 = t3.softmax(dim=-1)                      # Apply softmax to get attention weights
        t5 = t4.matmul(v)                            # Shape: (batch_size, num_heads, seq_len_q, head_dim)

        return t5

# Example usage
# Initialize the model
embed_dim = 64
num_heads = 8
model = AttentionModel(embed_dim, num_heads)

# Inputs to the model
batch_size = 2
seq_len = 10
attn_mask = torch.randn(batch_size, seq_len, seq_len)  # Random attention mask
query = torch.randn(batch_size, seq_len, embed_dim)     # Shape: (batch_size, seq_len, embed_dim)
key = torch.randn(batch_size, seq_len, embed_dim)       # Shape: (batch_size, seq_len, embed_dim)
value = torch.randn(batch_size, seq_len, embed_dim)     # Shape: (batch_size, seq_len, embed_dim)

# Forward pass
output = model(query, key, value, attn_mask)

# Print the shape of the output to verify
print(output.shape)  # Expected shape: (batch_size, num_heads, seq_len_q, head_dim)
