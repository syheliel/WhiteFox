import torch
import torch.nn as nn

class SelfAttentionModel(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttentionModel, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, attn_mask, inv_scale):
        q = self.queries(query).permute(0, 2, 1, 3)  # Permute the dimensions of the query tensor
        k = self.keys(key).permute(0, 2, 1, 3)  # Permute the dimensions of the key tensor
        v = self.values(value).permute(0, 2, 1, 3)  # Permute the dimensions of the value tensor
        
        bs = q.size(0)  # Get the size of the first dimension of the query tensor
        k_len = k.size(-2)  # Get the size of the second to last dimension of the key tensor
        
        scores = q @ k.transpose(-2, -1)  # Compute the dot product of the query and the transposed key tensor
        scores = scores.div(inv_scale)  # Divide the scores by the inverse scale
        
        fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)  # Create a tensor filled with negative infinity
        attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)  # Create an attention mask and expand it to the size of the scores tensor
        
        attention = torch.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1) @ v  # Apply softmax and compute the dot product with the value tensor
        return self.fc_out(attention)

# Initialize the model
embed_size = 64  # Embedding size
heads = 8  # Number of attention heads
model = SelfAttentionModel(embed_size, heads)

# Inputs to the model
batch_size = 1
seq_length = 10
query = torch.randn(batch_size, seq_length, embed_size)  # Query tensor
key = torch.randn(batch_size, seq_length, embed_size)    # Key tensor
value = torch.randn(batch_size, seq_length, embed_size)  # Value tensor
attn_mask = torch.zeros(batch_size, seq_length)           # Attention mask (assuming no masking)
inv_scale = embed_size ** 0.5  # Inverse scale for dot product

# Compute the output
output = model(query, key, value, attn_mask, inv_scale)

print(output.shape)  # Should print the shape of the output
