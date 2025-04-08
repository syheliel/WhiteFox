import torch
import torch.nn as nn

class SelfAttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, inv_scale=1.0):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_linear(query).permute(0, 2, 1, 3)
        k = self.key_linear(key).permute(0, 2, 1, 3)
        v = self.value_linear(value).permute(0, 2, 1, 3)

        # Get the size of the first dimension of the query tensor
        bs = q.size(0)
        # Get the size of the second to last dimension of the key tensor
        k_len = k.size(-2)

        # Compute the dot product of the query and the transposed key tensor
        scores = q @ k.transpose(-2, -1)
        scores = scores.div(inv_scale)

        # Create a tensor filled with negative infinity
        fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)

        # Create an attention mask and expand it to the size of the scores tensor
        if attn_mask is not None:
            attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
            scores = scores.masked_fill(attn_mask, fill_value)

        # Apply the softmax function to the scores tensor and compute the dot product with the value tensor
        attn_weights = torch.softmax(scores, dim=-1)
        output = attn_weights @ v
        
        return output

# Initializing the model
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
model = SelfAttentionModel(embed_dim, num_heads)

# Inputs to the model
batch_size = 2  # Example batch size
seq_length = 10 # Sequence length
x_query = torch.randn(batch_size, seq_length, embed_dim)
x_key = torch.randn(batch_size, seq_length, embed_dim)
x_value = torch.randn(batch_size, seq_length, embed_dim)
attn_mask = torch.randint(0, 2, (batch_size, seq_length)).to(torch.bool)  # Random attention mask

# Forward pass
output = model(x_query, x_key, x_value, attn_mask)

# Output shape
print(output.shape)
