import torch

class SelfAttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, inv_scale=1.0):
        q = self.query(query).permute(0, 2, 1, 3)  # Permute the dimensions of the query tensor
        k = self.key(key).permute(0, 2, 1, 3)      # Permute the dimensions of the key tensor
        v = self.value(value).permute(0, 2, 1, 3)  # Permute the dimensions of the value tensor
        
        bs = q.size(0)                 # Get the size of the first dimension of the query tensor
        k_len = k.size(-2)            # Get the size of the second to last dimension of the key tensor
        
        scores = q @ k.transpose(-2, -1)  # Compute the dot product of the query and the transposed key tensor
        scores = scores.div(inv_scale)      # Divide the scores by the inverse scale
        
        fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)  # Create a tensor filled with negative infinity
        
        if attn_mask is not None:
            attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)  # Expand the attention mask to the size of the scores tensor
            scores = scores.masked_fill(attn_mask, fill_value)  # Apply the attention mask
        
        attn_weights = torch.softmax(scores, dim=-1)  # Apply softmax to the scores tensor
        output = attn_weights @ v  # Compute the dot product with the value tensor
        
        return output.permute(0, 2, 1, 3)  # Permute back to the original shape

# Initializing the model
embed_dim = 64
num_heads = 8
model = SelfAttentionModel(embed_dim, num_heads)

# Generating input tensors
batch_size = 1
seq_length = 10

query = torch.randn(batch_size, seq_length, embed_dim)
key = torch.randn(batch_size, seq_length, embed_dim)
value = torch.randn(batch_size, seq_length, embed_dim)
attn_mask = torch.randint(0, 2, (batch_size, seq_length))

# Running the model
output = model(query, key, value, attn_mask)

print("Output shape:", output.shape)
