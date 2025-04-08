import torch

class SelfAttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionModel, self).__init__()
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.inv_scale = embed_dim ** -0.5

    def forward(self, query, key, value, attn_mask):
        q = self.query(query).view(query.size(0), -1, self.num_heads, query.size(2)).permute(0, 2, 1, 3)
        k = self.key(key).view(key.size(0), -1, self.num_heads, key.size(2)).permute(0, 2, 1, 3)
        v = self.value(value).view(value.size(0), -1, self.num_heads, value.size(2)).permute(0, 2, 1, 3)

        bs = q.size(0)
        k_len = k.size(-2)
        
        scores = q @ k.transpose(-2, -1)  # Dot product of query and transposed key
        scores = scores.div(self.inv_scale)  # Scale scores
        
        fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)
        attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
        
        attn_weights = torch.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1) @ v  # Apply softmax and weighted sum
        return attn_weights

# Initialize the model
embed_dim = 64  # Example embedding dimension
num_heads = 8  # Example number of attention heads
model = SelfAttentionModel(embed_dim, num_heads)

# Generate input tensors
batch_size = 2
seq_length = 10

# Input tensors
query = torch.randn(batch_size, seq_length, embed_dim)
key = torch.randn(batch_size, seq_length, embed_dim)
value = torch.randn(batch_size, seq_length, embed_dim)
attn_mask = torch.zeros(batch_size, seq_length)  # Example attention mask

# Forward pass
output = model(query, key, value, attn_mask)

print("Output shape:", output.shape)  # Output shape should be [batch_size, num_heads, seq_length, embed_dim/num_heads]
