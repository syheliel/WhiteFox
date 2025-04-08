import torch

class SelfAttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super(SelfAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        
        # Define linear layers for query, key, and value projections
        self.query_projection = torch.nn.Linear(embed_dim, embed_dim)
        self.key_projection = torch.nn.Linear(embed_dim, embed_dim)
        self.value_projection = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask, inv_scale):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_projection(query).permute(0, 2, 1, 3)
        k = self.key_projection(key).permute(0, 2, 1, 3)
        v = self.value_projection(value).permute(0, 2, 1, 3)

        bs = q.size(0)  # Get the size of the first dimension of the query tensor
        k_len = k.size(-2)  # Get the size of the second to last dimension of the key tensor

        # Compute the dot product of the query tensor and the transposed key tensor
        scores = q @ k.transpose(-2, -1)
        scores = scores.div(inv_scale)  # Divide the scores by the inverse scale

        # Create a tensor filled with negative infinity
        fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)

        # Create an attention mask
        attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)

        # Apply dropout to the softmax of the scores
        output = torch.nn.functional.dropout(
            torch.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1), self.dropout_p
        ) @ v  # Compute the dot product with the value tensor

        return output

# Example usage:
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
dropout_p = 0.1 # Dropout probability

# Initialize the model
model = SelfAttentionModel(embed_dim, num_heads, dropout_p)

# Create input tensors
batch_size = 2
seq_length = 10

# Random input tensors for query, key, value, and attention mask
query = torch.randn(batch_size, seq_length, embed_dim)  # (batch_size, seq_length, embed_dim)
key = torch.randn(batch_size, seq_length, embed_dim)    # (batch_size, seq_length, embed_dim)
value = torch.randn(batch_size, seq_length, embed_dim)  # (batch_size, seq_length, embed_dim)
attn_mask = torch.randint(0, 2, (batch_size, seq_length)).bool()  # (batch_size, seq_length)
inv_scale = embed_dim ** -0.5  # Inverse scale for the attention scores

# Forward pass
output = model(query, key, value, attn_mask, inv_scale)

print(output.shape)  # Output shape should be (batch_size, seq_length, embed_dim)
