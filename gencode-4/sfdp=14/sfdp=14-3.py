import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, inv_scale=1.0):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_linear(query).permute(0, 2, 1, 3)  # Shape: (bs, num_heads, seq_len_q, depth)
        k = self.key_linear(key).permute(0, 2, 1, 3)      # Shape: (bs, num_heads, seq_len_k, depth)
        v = self.value_linear(value).permute(0, 2, 1, 3)  # Shape: (bs, num_heads, seq_len_v, depth)

        bs = q.size(0)                                   # Batch size
        k_len = k.size(-2)                              # Length of the keys

        # Compute the dot product of the query and the transposed key tensor
        scores = q @ k.transpose(-2, -1)                # Shape: (bs, num_heads, seq_len_q, seq_len_k)
        scores = scores.div(inv_scale)                   # Divide the scores by the inverse scale

        # Create a tensor filled with negative infinity
        fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)

        # Create an attention mask and expand it to the size of the scores tensor
        if attn_mask is not None:
            attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)

        # Apply the softmax function to the scores tensor, fill the masked values with negative infinity, and compute the dot product with the value tensor
        attn_weights = torch.softmax(scores.masked_fill(attn_mask, fill_value) if attn_mask is not None else scores, dim=-1)
        output = attn_weights @ v                            # Shape: (bs, num_heads, seq_len_q, depth)

        return output

# Initializing the model
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
model = AttentionModel(embed_dim, num_heads)

# Inputs to the model
batch_size = 2
seq_len = 10
query_tensor = torch.randn(batch_size, seq_len, embed_dim)  # Shape: (batch_size, seq_len_q, embed_dim)
key_tensor = torch.randn(batch_size, seq_len, embed_dim)    # Shape: (batch_size, seq_len_k, embed_dim)
value_tensor = torch.randn(batch_size, seq_len, embed_dim)  # Shape: (batch_size, seq_len_v, embed_dim)
attn_mask = torch.randint(0, 2, (batch_size, seq_len, seq_len)).to(torch.bool)  # Attention mask

# Forward pass
output = model(query_tensor, key_tensor, value_tensor, attn_mask)

# Output shape
print(output.shape)  # Expected shape: (batch_size, num_heads, seq_len, depth)
