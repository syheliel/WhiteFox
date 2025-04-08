import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_size, heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.dropout_p = dropout_p
        self.scale_factor = embed_size ** -0.5  # Scaling factor for dot product
        
        # Define linear layers for query, key, and value
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        
    def forward(self, query, key, value):
        # Compute the dot product of the query and the transposed key
        qk = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale the dot product by a factor
        scaled_qk = qk.mul(self.scale_factor)
        
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        
        # Apply dropout to the softmax output
        dropout_qk = nn.functional.dropout(softmax_qk, p=self.dropout_p, training=self.training)
        
        # Compute the dot product of the dropout output and the value
        output = dropout_qk.matmul(value)
        
        return output

# Example of initializing the model
embed_size = 64  # Size of the embedding
heads = 8       # Number of attention heads
dropout_p = 0.1 # Dropout probability
model = AttentionModel(embed_size, heads, dropout_p)

# Generating an input tensor
# Example tensor for query, key, and value with batch size 1 and sequence length 10
query = torch.randn(1, 10, embed_size)  # Shape: (batch_size, seq_length, embed_size)
key = torch.randn(1, 10, embed_size)    # Shape: (batch_size, seq_length, embed_size)
value = torch.randn(1, 10, embed_size)  # Shape: (batch_size, seq_length, embed_size)

# Forward pass through the model
output = model(query, key, value)

# Output shape
print(output.shape)  # Should be (1, 10, embed_size) same as value tensor shape
