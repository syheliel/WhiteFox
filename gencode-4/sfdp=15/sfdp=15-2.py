import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super(SelfAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask, inv_scale):
        q = self.query(query).permute(0, 2, 1, 3)  # Permute the dimensions of the query tensor
        k = self.key(key).permute(0, 2, 1, 3)      # Permute the dimensions of the key tensor
        v = self.value(value).permute(0, 2, 1, 3)  # Permute the dimensions of the value tensor

        attn_weights = (torch.matmul(q, k.transpose(-2, -1)).div(inv_scale) + attn_mask).softmax(dim=-1)  # Compute the attention weights
        dropout_attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)  # Apply dropout to the attention weights
        output = dropout_attn_weights.matmul(v)  # Multiply the dropout attention weights by the value tensor
        
        return output

# Example usage
embed_dim = 64
num_heads = 8
dropout_p = 0.1
model = SelfAttentionModel(embed_dim, num_heads, dropout_p)

# Generating random input tensors
batch_size = 2
seq_length = 10
query_tensor = torch.randn(batch_size, seq_length, embed_dim)
key_tensor = torch.randn(batch_size, seq_length, embed_dim)
value_tensor = torch.randn(batch_size, seq_length, embed_dim)
attn_mask = torch.zeros(batch_size, 1, seq_length, seq_length)  # Example attention mask
inv_scale = torch.tensor(1.0)  # Example scale factor

# Forward pass
output = model(query_tensor, key_tensor, value_tensor, attn_mask, inv_scale)
