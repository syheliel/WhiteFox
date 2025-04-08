import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, attn_mask, inv_scale):
        q = self.query_proj(query).view(query.size(0), -1, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
        k = self.key_proj(key).view(key.size(0), -1, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
        v = self.value_proj(value).view(value.size(0), -1, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
        
        bs = q.size(0)
        k_len = k.size(-2)
        
        scores = q @ k.transpose(-2, -1)
        scores = scores.div(inv_scale)
        
        fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)
        attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
        
        output = F.dropout(
            F.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1), self.dropout_p
        ) @ v
        
        return output

# Initializing the model
embed_dim = 64
num_heads = 8
dropout_p = 0.1
model = AttentionModel(embed_dim, num_heads, dropout_p)

# Inputs to the model
batch_size = 2
seq_length = 10  # Sequence length for query, key, and value
query = torch.randn(batch_size, seq_length, embed_dim)
key = torch.randn(batch_size, seq_length, embed_dim)
value = torch.randn(batch_size, seq_length, embed_dim)
attn_mask = torch.randint(0, 2, (batch_size, seq_length)).to(torch.bool)  # Random attention mask
inv_scale = torch.tensor(embed_dim ** 0.5)  # Inverse scale for attention

# Forward pass
output = model(query, key, value, attn_mask, inv_scale)
print(output.shape)  # Should be [batch_size, num_heads, seq_length, embed_dim // num_heads]
