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
        q = self.query(query).permute(0, 2, 1, 3)  # Shape: (batch_size, num_heads, seq_length, head_dim)
        k = self.key(key).permute(0, 2, 1, 3)      # Shape: (batch_size, num_heads, seq_length, head_dim)
        v = self.value(value).permute(0, 2, 1, 3)  # Shape: (batch_size, num_heads, seq_length, head_dim)

        bs = q.size(0)         # Get the size of the first dimension of the query tensor
        k_len = k.size(-2)     # Get the size of the second to last dimension of the key tensor
        
        # Compute the dot product of the query and the transposed key tensor
        scores = q @ k.transpose(-2, -1)
        scores = scores.div(inv_scale)  # Divide the scores by the inverse scale
        
        # Create a tensor filled with negative infinity
        fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)
        
        # Create an attention mask and expand it to the size of the scores tensor
        if attn_mask is not None:
            attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
            scores = scores.masked_fill(attn_mask, fill_value)
        
        # Apply the softmax function to the scores and compute the dot product with the value tensor
        attention_weights = torch.softmax(scores, dim=-1)
        output = attention_weights @ v
        
        return output

# Initialize the model
embed_dim = 64
num_heads = 8
model = SelfAttentionModel(embed_dim, num_heads)

# Generate input tensors
batch_size = 1
seq_length = 10  # Length of the sequence for the queries, keys, and values
query_tensor = torch.randn(batch_size, seq_length, embed_dim)
key_tensor = torch.randn(batch_size, seq_length, embed_dim)
value_tensor = torch.randn(batch_size, seq_length, embed_dim)
attn_mask = torch.zeros(batch_size, seq_length)  # Example attention mask

# Forward pass
output = model(query_tensor, key_tensor, value_tensor, attn_mask)

print("Output shape:", output.shape)  # Output shape should be (batch_size, seq_length, embed_dim)
