import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, dropout_p):
        super(AttentionModel, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x):
        # Assuming x has shape (batch_size, seq_length, embed_dim)
        q = self.query(x)  # Query
        k = self.key(x)    # Key
        v = self.value(x)  # Value

        # Compute the dot product of the query and the transposed key
        qk = torch.matmul(q, k.transpose(-2, -1))
        
        # Scale the dot product by a factor
        scale_factor = q.size(-1) ** 0.5  # Scaling factor (sqrt of embedding dimension)
        scaled_qk = qk.mul(scale_factor)
        
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        
        # Apply dropout to the softmax output
        dropout_qk = self.dropout(softmax_qk)
        
        # Compute the dot product of the dropout output and the value
        output = dropout_qk.matmul(v)
        
        return output

# Initializing the model
embed_dim = 64  # Embedding dimension
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_dim, dropout_p)

# Inputs to the model
batch_size = 1
seq_length = 10
input_tensor = torch.randn(batch_size, seq_length, embed_dim)

# Forward pass
output = model(input_tensor)
