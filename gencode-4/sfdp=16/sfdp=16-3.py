import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super(SelfAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_layer(query).permute(0, 2, 1, 3)
        k = self.key_layer(key).permute(0, 2, 1, 3)
        v = self.value_layer(value).permute(0, 2, 1, 3)

        bs = q.size(0)  # Get the batch size
        k_len = k.size(-2)  # Get the size of the second to last dimension of the key tensor

        # Compute the dot product of the query tensor and the transposed key tensor
        scores = q @ k.transpose(-2, -1)
        inv_scale = self.embed_dim ** 0.5  # Inverse scale
        scores = scores.div(inv_scale)  # Divide the scores by the inverse scale

        fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)  # Create a tensor filled with negative infinity
        attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)  # Create an attention mask

        # Apply dropout to the softmax of the scores, mask the scores with the attention mask
        output = F.dropout(
            F.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1), self.dropout_p
        ) @ v  # Compute the dot product with the value tensor

        return output

# Initialization of the model
embed_dim = 64  # Example embedding dimension
num_heads = 8   # Example number of attention heads
dropout_p = 0.1  # Dropout probability
model = SelfAttentionModel(embed_dim, num_heads, dropout_p)

# Generating input tensors for the model
batch_size = 2
seq_length = 10
query = torch.randn(batch_size, seq_length, embed_dim)  # Query tensor
key = torch.randn(batch_size, seq_length, embed_dim)    # Key tensor
value = torch.randn(batch_size, seq_length, embed_dim)  # Value tensor
attn_mask = torch.zeros(batch_size, seq_length)         # Attention mask (0s where attention is allowed)

# Forward pass through the model
output = model(query, key, value, attn_mask)
print(output.shape)  # Output shape
