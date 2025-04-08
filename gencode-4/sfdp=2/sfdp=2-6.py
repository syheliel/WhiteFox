import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        
        # Define linear layers for query, key, and value
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, inv_scale_factor):
        # Compute the dot product of query and key
        qk = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale the dot product by the inverse scale factor
        scaled_qk = qk.div(inv_scale_factor)
        
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        
        # Apply dropout to the softmax output
        dropout_qk = nn.functional.dropout(softmax_qk, p=self.dropout_p, training=self.training)
        
        # Compute the dot product of the dropout output and the value tensor
        output = dropout_qk.matmul(value)
        
        return output

# Initialize the model with appropriate parameters
embed_dim = 64  # Example embedding dimension
num_heads = 8   # Example number of heads for multi-head attention
dropout_p = 0.1 # Dropout probability
model = AttentionModel(embed_dim, num_heads, dropout_p)

# Create input tensors for query, key, and value
batch_size = 1
seq_length = 10  # Example sequence length

query = torch.randn(batch_size, seq_length, embed_dim)
key = torch.randn(batch_size, seq_length, embed_dim)
value = torch.randn(batch_size, seq_length, embed_dim)
inv_scale_factor = embed_dim ** 0.5  # Inverse scale factor for scaling

# Forward pass through the model
output = model(query, key, value, inv_scale_factor)

# Print output shape
print(output.shape)  # Should be (batch_size, seq_length, embed_dim)
