import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, dropout_p):
        super(AttentionModel, self).__init__()
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.scale_factor = 1.0 / (embed_dim ** 0.5)  # Scale factor for dot product

    def forward(self, query, key, value):
        # Compute the dot product of the query and the transposed key
        qk = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale the dot product by a factor
        scaled_qk = qk.mul(self.scale_factor)
        
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        
        # Apply dropout to the softmax output
        dropout_qk = self.dropout(softmax_qk)
        
        # Compute the dot product of the dropout output and the value
        output = dropout_qk.matmul(value)
        
        return output

# Initialize the model
embed_dim = 64  # Embedding dimension
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_dim, dropout_p)

# Inputs to the model
batch_size = 1
seq_length = 10  # Example sequence length
query = torch.randn(batch_size, seq_length, embed_dim)
key = torch.randn(batch_size, seq_length, embed_dim)
value = torch.randn(batch_size, seq_length, embed_dim)

# Forward pass
output = model(query, key, value)
print(output)
