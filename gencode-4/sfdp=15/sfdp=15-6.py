import torch
import torch.nn as nn

class SelfAttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super(SelfAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        
        # Linear layers for query, key, and value
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, attn_mask, inv_scale):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_layer(query).permute(0, 2, 1, 3)  # Shape: [batch_size, num_heads, seq_len, head_dim]
        k = self.key_layer(key).permute(0, 2, 1, 3)      # Shape: [batch_size, num_heads, seq_len, head_dim]
        v = self.value_layer(value).permute(0, 2, 1, 3)  # Shape: [batch_size, num_heads, seq_len, head_dim]
        
        # Compute the attention weights
        attn_weights = (torch.matmul(q, k.transpose(-2, -1)).div(inv_scale) + attn_mask).softmax(dim=-1)
        
        # Apply dropout to the attention weights
        dropout_attn_weights = nn.functional.dropout(attn_weights, p=self.dropout_p, training=self.training)
        
        # Multiply the dropout attention weights by the value tensor
        output = dropout_attn_weights.matmul(v)  # Shape: [batch_size, num_heads, seq_len, head_dim]
        
        return output

# Hyperparameters
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
dropout_p = 0.1 # Dropout probability
seq_len = 10    # Sequence length
batch_size = 1  # Batch size

# Initialize the model
model = SelfAttentionModel(embed_dim, num_heads, dropout_p)

# Inputs to the model
query = torch.randn(batch_size, seq_len, embed_dim)  # Shape: [batch_size, seq_len, embed_dim]
key = torch.randn(batch_size, seq_len, embed_dim)    # Shape: [batch_size, seq_len, embed_dim]
value = torch.randn(batch_size, seq_len, embed_dim)  # Shape: [batch_size, seq_len, embed_dim]
attn_mask = torch.zeros(batch_size, num_heads, seq_len, seq_len)  # Attention mask
inv_scale = torch.tensor(1.0)  # Inverse scale for attention computation

# Forward pass through the model
output = model(query, key, value, attn_mask, inv_scale)

print("Output shape:", output.shape)
