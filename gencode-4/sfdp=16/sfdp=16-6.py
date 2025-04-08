import torch
import torch.nn as nn

class SelfAttentionModel(nn.Module):
    def __init__(self, embed_size, heads, dropout_p):
        super(SelfAttentionModel, self).__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.dropout_p = dropout_p
        
        self.query_layer = nn.Linear(embed_size, embed_size)
        self.key_layer = nn.Linear(embed_size, embed_size)
        self.value_layer = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, attn_mask, inv_scale):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_layer(query).permute(0, 2, 1, 3)  # (batch_size, heads, seq_length, embed_size/heads)
        k = self.key_layer(key).permute(0, 2, 1, 3)      # (batch_size, heads, seq_length, embed_size/heads)
        v = self.value_layer(value).permute(0, 2, 1, 3)  # (batch_size, heads, seq_length, embed_size/heads)

        bs = q.size(0)  # Get the size of the first dimension of the query tensor
        k_len = k.size(-2)  # Get the size of the second to last dimension of the key tensor

        # Compute the dot product of the query tensor and the transposed key tensor
        scores = q @ k.transpose(-2, -1)
        scores = scores.div(inv_scale)  # Divide the scores by the inverse scale

        fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)  # Create a tensor filled with negative infinity
        attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)  # Create an attention mask

        output = nn.functional.dropout(
            torch.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1), self.dropout_p
        ) @ v  # Apply dropout to the softmax of the scores and compute the dot product with the value tensor

        return output

# Initializing the model
embed_size = 64   # Size of the embedding
heads = 8        # Number of attention heads
dropout_p = 0.1  # Dropout probability
model = SelfAttentionModel(embed_size, heads, dropout_p)

# Generate input tensors
batch_size = 1
seq_length = 10  # Sequence length

# Create random input tensors for query, key, value, and attention mask
query = torch.randn(batch_size, seq_length, embed_size)
key = torch.randn(batch_size, seq_length, embed_size)
value = torch.randn(batch_size, seq_length, embed_size)
attn_mask = torch.zeros(batch_size, seq_length)  # Attention mask (0 means valid, 1 means masked)
inv_scale = float(embed_size) ** 0.5  # Inverse scale for the scores

# Forward pass through the model
output = model(query, key, value, attn_mask, inv_scale)

# Output tensor
print("Output shape:", output.shape)  # Should be (batch_size, heads, seq_length, embed_size/heads)
