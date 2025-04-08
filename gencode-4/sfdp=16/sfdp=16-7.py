import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.query_projection = torch.nn.Linear(embed_dim, embed_dim)
        self.key_projection = torch.nn.Linear(embed_dim, embed_dim)
        self.value_projection = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask, inv_scale):
        q = self.query_projection(query).permute(0, 2, 1, 3)  # Permute the query dimensions
        k = self.key_projection(key).permute(0, 2, 1, 3)      # Permute the key dimensions
        v = self.value_projection(value).permute(0, 2, 1, 3)  # Permute the value dimensions

        bs = q.size(0)  # Batch size
        k_len = k.size(-2)  # Length of the key sequence

        scores = q @ k.transpose(-2, -1)  # Dot product for attention scores
        scores = scores.div(inv_scale)  # Scale the scores

        fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)  # Negative infinity tensor
        attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)  # Expand attention mask

        # Apply dropout to the softmax of the scores
        output = torch.nn.functional.dropout(
            torch.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1), 
            p=self.dropout_p
        ) @ v  # Compute output

        return output

# Initialize the model with parameters
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
dropout_p = 0.1 # Dropout probability
model = AttentionModel(embed_dim, num_heads, dropout_p)

# Create input tensors
batch_size = 2
seq_len = 10
input_dim = embed_dim

query = torch.randn(batch_size, seq_len, input_dim)
key = torch.randn(batch_size, seq_len, input_dim)
value = torch.randn(batch_size, seq_len, input_dim)
attn_mask = torch.zeros(batch_size, seq_len)  # No masking in this example
inv_scale = embed_dim ** 0.5  # Inverse scale for the dot product

# Get output from the model
output = model(query, key, value, attn_mask, inv_scale)

print("Output shape:", output.shape)  # Output shape after attention mechanism
