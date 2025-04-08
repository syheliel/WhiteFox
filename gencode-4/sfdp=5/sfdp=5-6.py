import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, dropout_p):
        super(AttentionModel, self).__init__()
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value, attn_mask):
        # Compute the dot product of the query and key, and scale it
        qk = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        
        # Add the attention mask to the scaled dot product
        qk = qk + attn_mask
        
        # Apply softmax to the result
        attn_weight = F.softmax(qk, dim=-1)
        
        # Apply dropout to the softmax output
        attn_weight = self.dropout(attn_weight)
        
        # Compute the dot product of the dropout output and the value
        output = torch.matmul(attn_weight, value)
        
        return output

# Example of initializing the model
embed_dim = 64  # Size of the embeddings
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_dim, dropout_p)

# Generate input tensors
batch_size = 2  # Number of samples in a batch
seq_len = 10    # Length of the sequence

# Random input tensors for query, key, and value
query = torch.randn(batch_size, seq_len, embed_dim)
key = torch.randn(batch_size, seq_len, embed_dim)
value = torch.randn(batch_size, seq_len, embed_dim)

# Attention mask (e.g., for padding or look-ahead)
attn_mask = torch.zeros(batch_size, seq_len, seq_len)  # No mask applied; can be adjusted as needed

# Forward pass through the model
output = model(query, key, value, attn_mask)

# Output shape
print(output.shape)  # Should be of shape (batch_size, seq_len, embed_dim)
