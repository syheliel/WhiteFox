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

    def forward(self, x, attn_mask=None, inv_scale=1.0):
        # Assuming x has shape (batch_size, seq_length, embed_dim)
        bs, seq_len, _ = x.size()
        
        q = self.query(x).view(bs, seq_len, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)  # (bs, num_heads, seq_len, head_dim)
        k = self.key(x).view(bs, seq_len, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)  # (bs, num_heads, seq_len, head_dim)
        v = self.value(x).view(bs, seq_len, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)  # (bs, num_heads, seq_len, head_dim)

        k_len = k.size(-2)  # Size of the key dimension (sequence length)

        scores = q @ k.transpose(-2, -1)  # Dot product attention scores
        scores = scores.div(inv_scale)  # Scale the scores
        
        fill_value = torch.full((), -float("inf"), dtype=x.dtype, device=x.device)  # Tensor filled with negative infinity

        if attn_mask is not None:
            attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)  # Expand attention mask to match scores dimensions
            scores = scores.masked_fill(attn_mask, fill_value)  # Apply attention mask
        
        attn_weights = torch.softmax(scores, dim=-1)  # Softmax over the scores
        output = attn_weights @ v  # Contextual output

        return output

# Initializing the model
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
model = SelfAttentionModel(embed_dim, num_heads)

# Inputs to the model
batch_size = 2
seq_length = 10
x = torch.randn(batch_size, seq_length, embed_dim)  # Random input tensor
attn_mask = torch.zeros(batch_size, seq_length)  # Attention mask (can be adjusted as needed)
inv_scale = 1.0  # Inverse scale for softmax

# Forward pass
output = model(x, attn_mask, inv_scale)

print(output.shape)  # Output shape
