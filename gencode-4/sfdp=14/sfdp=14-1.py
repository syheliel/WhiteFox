import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, inv_scale=1.0):
        q = self.query(query).permute(0, 2, 1, 3)  # Permute the dimensions of the query tensor
        k = self.key(key).permute(0, 2, 1, 3)      # Permute the dimensions of the key tensor
        v = self.value(value).permute(0, 2, 1, 3)  # Permute the dimensions of the value tensor
        
        bs = q.size(0)                             # Get the size of the first dimension of the query tensor
        k_len = k.size(-2)                        # Get the size of the second to last dimension of the key tensor
        
        scores = q @ k.transpose(-2, -1)          # Compute the dot product of the query and the transposed key tensor
        scores = scores.div(inv_scale)             # Divide the scores by the inverse scale
        
        fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)  # Create a tensor filled with negative infinity
        if attn_mask is not None:
            attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)  # Create an attention mask and expand it to the size of the scores tensor
            scores = scores.masked_fill(attn_mask, fill_value)  # Fill the masked values with negative infinity
            
        attention_weights = torch.softmax(scores, dim=-1)  # Apply the softmax function to the scores tensor
        output = attention_weights @ v  # Compute the dot product with the value tensor
        
        return output

# Initialize the model
embed_dim = 64
head_dim = 16
model = AttentionModel(embed_dim, head_dim)

# Inputs to the model
batch_size = 2
seq_length = 10
query = torch.randn(batch_size, seq_length, embed_dim)
key = torch.randn(batch_size, seq_length, embed_dim)
value = torch.randn(batch_size, seq_length, embed_dim)
attn_mask = torch.randint(0, 2, (batch_size, seq_length))  # Random attention mask

# Forward pass
output = model(query, key, value, attn_mask)

# Output shape
print(output.shape)  # Should be (batch_size, seq_length, embed_dim)
