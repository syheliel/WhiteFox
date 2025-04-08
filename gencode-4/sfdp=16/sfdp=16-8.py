import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.query_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.key_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.value_linear = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, inv_scale=1.0):
        q = self.query_linear(query).permute(0, 2, 1, 3)  # Permute dimensions
        k = self.key_linear(key).permute(0, 2, 1, 3)      # Permute dimensions
        v = self.value_linear(value).permute(0, 2, 1, 3)  # Permute dimensions
        
        bs = q.size(0)  # Batch size
        k_len = k.size(-2)  # Length of key sequences
        scores = q @ k.transpose(-2, -1)  # Dot product
        scores = scores.div(inv_scale)  # Scale scores

        fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)  # Negative infinity tensor
        if attn_mask is not None:
            attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)  # Create attention mask
            scores = scores.masked_fill(attn_mask, fill_value)  # Apply attention mask

        attn_weights = torch.nn.functional.dropout(
            torch.softmax(scores, dim=-1), self.dropout_p
        )
        output = attn_weights @ v  # Final output
        return output

# Initializing the model
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
dropout_p = 0.1 # Dropout probability
model = AttentionModel(embed_dim, num_heads, dropout_p)

# Inputs to the model
batch_size = 2
seq_length = 10
query = torch.randn(batch_size, seq_length, embed_dim)
key = torch.randn(batch_size, seq_length, embed_dim)
value = torch.randn(batch_size, seq_length, embed_dim)
attn_mask = torch.randint(0, 2, (batch_size, seq_length))  # Random attention mask (0 or 1)

# Forward pass
output = model(query, key, value, attn_mask)

print(output.shape)  # Output shape
