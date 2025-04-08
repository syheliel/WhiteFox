import torch
import torch.nn as nn

class SelfAttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, inv_scale=1.0):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query(query).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        k = self.key(key).permute(0, 2, 1, 3)      # (batch_size, num_heads, seq_len, head_dim)
        v = self.value(value).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

        bs = q.size(0)  # Get batch size
        k_len = k.size(-2)  # Get the length of the keys

        # Compute dot product of query and transposed key tensor
        scores = q @ k.transpose(-2, -1)  # (batch_size, num_heads, seq_len, seq_len)
        scores = scores.div(inv_scale)  # Divide scores by inverse scale

        # Create a tensor filled with negative infinity
        fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)

        # Create an attention mask and expand it to the size of the scores tensor
        if attn_mask is not None:
            attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
            scores = scores.masked_fill(attn_mask, fill_value)

        # Apply softmax and compute the dot product with the value tensor
        attention_weights = torch.softmax(scores, dim=-1)
        output = attention_weights @ v  # (batch_size, num_heads, seq_len, head_dim)

        return output

# Initializing the model
embed_dim = 64  # Example embedding dimension
num_heads = 8   # Example number of attention heads
model = SelfAttentionModel(embed_dim, num_heads)

# Inputs to the model
batch_size = 2   # Example batch size
seq_len = 10     # Example sequence length
x_query = torch.randn(batch_size, seq_len, embed_dim)  # Random query tensor
x_key = torch.randn(batch_size, seq_len, embed_dim)    # Random key tensor
x_value = torch.randn(batch_size, seq_len, embed_dim)  # Random value tensor
attn_mask = torch.zeros(batch_size, seq_len)  # Example attention mask (can be customized)
inv_scale = embed_dim ** -0.5  # Example inverse scale

# Forward pass
output = model(x_query, x_key, x_value, attn_mask, inv_scale)

print(output.shape)  # Output shape will be (batch_size, num_heads, seq_len, head_dim)
