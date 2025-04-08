import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        
        # Define linear layers for query, key, and value projections
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, attn_mask=None):
        # Project the inputs to the query, key, and value space
        q = self.query_linear(query).view(query.shape[0], -1, self.num_heads, self.embed_size // self.num_heads).permute(0, 2, 1, 3)
        k = self.key_linear(key).view(key.shape[0], -1, self.num_heads, self.embed_size // self.num_heads).permute(0, 2, 1, 3)
        v = self.value_linear(value).view(value.shape[0], -1, self.num_heads, self.embed_size // self.num_heads).permute(0, 2, 1, 3)
        
        # Compute attention weights
        inv_scale = (self.embed_size // self.num_heads) ** 0.5
        attn_weights = (torch.matmul(q, k.transpose(-2, -1)) / inv_scale)  # Scale the dot product
        if attn_mask is not None:
            attn_weights += attn_mask  # Add attention mask if provided
        attn_weights = F.softmax(attn_weights, dim=-1)  # Apply softmax to get probabilities
        
        # Apply dropout
        dropout_attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
        
        # Compute output
        output = dropout_attn_weights.matmul(v)  # Multiply by value tensor
        return output

# Initializing the model
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_size, num_heads, dropout_p)

# Generate input tensors
batch_size = 2
sequence_length = 10  # Length of the input sequences
query = torch.randn(batch_size, sequence_length, embed_size)
key = torch.randn(batch_size, sequence_length, embed_size)
value = torch.randn(batch_size, sequence_length, embed_size)

# Mask (optional)
attn_mask = torch.zeros(batch_size, num_heads, sequence_length, sequence_length)  # Example mask

# Forward pass
output = model(query, key, value, attn_mask)
print(output.shape)  # Output shape should be (batch_size, num_heads, sequence_length, embed_size // num_heads)
