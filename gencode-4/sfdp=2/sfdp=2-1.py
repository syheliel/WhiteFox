import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, dropout_p):
        super(AttentionModel, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.inv_scale_factor = embed_dim ** 0.5  # Inverse scale factor

    def forward(self, x):
        q = self.query(x)  # Linear transformation for query
        k = self.key(x)    # Linear transformation for key
        v = self.value(x)  # Linear transformation for value
        
        qk = torch.matmul(q, k.transpose(-2, -1))  # Compute the dot product
        scaled_qk = qk.div(self.inv_scale_factor)  # Scale the dot product
        softmax_qk = scaled_qk.softmax(dim=-1)      # Apply softmax to scaled dot product
        dropout_qk = self.dropout(softmax_qk)       # Apply dropout
        output = dropout_qk.matmul(v)               # Compute the dot product with value
        
        return output

# Initializing the model
embed_dim = 64  # Embedding dimension
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_dim, dropout_p)

# Inputs to the model
batch_size = 1
sequence_length = 10
x_input = torch.randn(batch_size, sequence_length, embed_dim)  # Random input tensor

# Forward pass
output = model(x_input)
print(output.shape)  # Output shape
