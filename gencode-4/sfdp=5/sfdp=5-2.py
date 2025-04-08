import torch
import torch.nn as nn
import math

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value, attn_mask=None):
        # Compute the dot product of the query and key, and scale it
        qk = self.query_linear(query) @ self.key_linear(key).transpose(-2, -1) / math.sqrt(self.embed_size)
        
        # Add the attention mask to the scaled dot product
        if attn_mask is not None:
            qk += attn_mask
        
        # Apply softmax to the result
        attn_weight = torch.softmax(qk, dim=-1)
        
        # Apply dropout to the softmax output
        attn_weight = self.dropout(attn_weight)
        
        # Compute the dot product of the dropout output and the value
        output = attn_weight @ self.value_linear(value)
        
        return output

# Initialize model parameters
embed_size = 64  # Size of the embedding vector
num_heads = 8    # Number of attention heads
dropout_p = 0.1  # Dropout probability

# Create an instance of the model
model = AttentionModel(embed_size, num_heads, dropout_p)

# Generate input tensors for the model
batch_size = 1
seq_length = 10  # Length of the input sequence
query = torch.randn(batch_size, seq_length, embed_size)  # Query tensor
key = torch.randn(batch_size, seq_length, embed_size)    # Key tensor
value = torch.randn(batch_size, seq_length, embed_size)  # Value tensor
attn_mask = torch.zeros(batch_size, seq_length, seq_length)  # Attention mask (optional)

# Forward pass
output = model(query, key, value, attn_mask)

# Print the output shape
print(output.shape)  # Expected shape: (batch_size, seq_length, embed_size)
