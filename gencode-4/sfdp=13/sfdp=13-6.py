import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionModel, self).__init__()
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.inv_scale = embed_dim ** -0.5  # Inverse scale for attention
        self.attn_mask = None  # Placeholder for attention mask

    def forward(self, query, key, value):
        # Permute dimensions
        q = self.q_linear(query).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        k = self.k_linear(key).permute(0, 2, 1, 3)    # (batch_size, num_heads, seq_len, head_dim)
        v = self.v_linear(value).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

        # Compute attention scores
        t1 = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)
        t2 = t1 * self.inv_scale  # Scale the scores
        if self.attn_mask is not None:
            t2 += self.attn_mask  # Add attention mask if provided
        t3 = t2.softmax(dim=-1)  # Apply softmax
        t4 = t3.matmul(v)  # Attention output

        return t4.permute(0, 2, 1, 3)  # Return to original shape

# Initializing the model
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
model = AttentionModel(embed_dim, num_heads)

# Generating the input tensors
batch_size = 1   # Number of examples in a batch
seq_len = 10     # Length of the sequence (number of tokens)
input_tensor = torch.randn(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)

# Forward pass through the model
output_tensor = model(input_tensor, input_tensor, input_tensor)
print(output_tensor.shape)  # Output shape
