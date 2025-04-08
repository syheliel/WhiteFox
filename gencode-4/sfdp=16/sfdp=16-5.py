import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionModel(nn.Module):
    def __init__(self, embed_size, heads, dropout_p):
        super(SelfAttentionModel, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.dropout_p = dropout_p
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, query, key, value, attn_mask, inv_scale):
        q = self.queries(query).permute(0, 2, 1, 3)  # Shape: (batch_size, heads, seq_length, head_dim)
        k = self.keys(key).permute(0, 2, 1, 3)      # Shape: (batch_size, heads, seq_length, head_dim)
        v = self.values(value).permute(0, 2, 1, 3)  # Shape: (batch_size, heads, seq_length, head_dim)

        bs = q.size(0)  # Batch size
        k_len = k.size(-2)  # Length of keys

        scores = q @ k.transpose(-2, -1)  # Shape: (batch_size, heads, seq_length, seq_length)
        scores = scores.div(inv_scale)  # Scale scores

        fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)
        attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)  # Expand mask
        output = F.dropout(
            F.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1), p=self.dropout_p
        ) @ v  # Apply dropout and compute output

        return self.fc_out(output.permute(0, 2, 1, 3))  # Reshape back to (batch_size, seq_length, embed_size)

# Initialize the model
embed_size = 64  # Size of the embedding
heads = 4  # Number of attention heads
dropout_p = 0.1  # Dropout probability
model = SelfAttentionModel(embed_size, heads, dropout_p)

# Example input tensors
batch_size = 2
seq_length = 10
query = torch.randn(batch_size, seq_length, embed_size)
key = torch.randn(batch_size, seq_length, embed_size)
value = torch.randn(batch_size, seq_length, embed_size)
attn_mask = torch.randint(0, 2, (batch_size, seq_length))  # Random attention mask
inv_scale = (embed_size ** 0.5)  # Inverse scale for scoring

# Forward pass through the model
output = model(query, key, value, attn_mask, inv_scale)

# Output from the model
print(output.shape)  # Should output shape: (batch_size, seq_length, embed_size)
