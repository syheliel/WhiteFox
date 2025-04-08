import torch

# Model
class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.query_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.key_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.value_linear = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_linear(query).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        k = self.key_linear(key).permute(0, 2, 1, 3)      # (batch_size, num_heads, seq_len, head_dim)
        v = self.value_linear(value).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

        # Compute the attention weights
        inv_scale = self.embed_dim ** 0.5
        attn_weights = (torch.matmul(q, k.transpose(-2, -1)).div(inv_scale) + (attn_mask if attn_mask is not None else 0)).softmax(dim=-1)

        # Apply dropout to the attention weights
        dropout_attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout_p)
        
        # Multiply the dropout attention weights by the value tensor
        output = dropout_attn_weights.matmul(v)  # (batch_size, num_heads, seq_len, head_dim)
        
        return output

# Initializing the model
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
dropout_p = 0.1 # Dropout probability
model = AttentionModel(embed_dim, num_heads, dropout_p)

# Inputs to the model
batch_size = 2
seq_len = 10

# Create random input tensors for query, key, and value
query = torch.randn(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)
key = torch.randn(batch_size, seq_len, embed_dim)    # (batch_size, seq_len, embed_dim)
value = torch.randn(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)
attn_mask = torch.zeros(batch_size, seq_len, seq_len)  # (batch_size, seq_len, seq_len)

# Forward pass through the model
output = model(query, key, value, attn_mask)

# Output result
print(output.shape)  # Should print (batch_size, num_heads, seq_len, head_dim)
