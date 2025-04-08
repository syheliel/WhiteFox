import torch

class SelfAttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super(SelfAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask):
        q = self.query(query).permute(0, 2, 1, 3)  # Permute the dimensions of the query tensor
        k = self.key(key).permute(0, 2, 1, 3)      # Permute the dimensions of the key tensor
        v = self.value(value).permute(0, 2, 1, 3)  # Permute the dimensions of the value tensor
        
        bs = q.size(0)                            # Get the size of the first dimension of the query tensor
        k_len = k.size(-2)                       # Get the size of the second to last dimension of the key tensor
        
        inv_scale = self.embed_dim ** -0.5       # Inverse scale for dot product
        scores = q @ k.transpose(-2, -1)         # Compute the dot product of q and k^T
        scores = scores.div(inv_scale)           # Divide the scores by the inverse scale
        
        fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)  # Tensor filled with -inf
        attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)  # Expand attention mask
        output = torch.nn.functional.dropout(
            torch.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1), self.dropout_p
        ) @ v                                     # Dot product with the value tensor
        
        return output

# Initializing the model
embed_dim = 64
num_heads = 8
dropout_p = 0.1
model = SelfAttentionModel(embed_dim, num_heads, dropout_p)

# Inputs to the model
batch_size = 1
seq_length = 10
attn_mask = torch.zeros(batch_size, seq_length)  # Attention mask
query = torch.randn(batch_size, seq_length, embed_dim)  # Random query tensor
key = torch.randn(batch_size, seq_length, embed_dim)    # Random key tensor
value = torch.randn(batch_size, seq_length, embed_dim)  # Random value tensor

# Forward pass
output = model(query, key, value, attn_mask)
print(output.shape)  # Output shape
