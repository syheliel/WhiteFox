import torch
import torch.nn as nn

# Define the model
class AttentionModel(nn.Module):
    def __init__(self, dim, dropout_p=0.1):
        super().__init__()
        self.dim = dim
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, query, key, value, inv_scale_factor):
        q = query.permute(0, 2, 1, 3)  # Permute the dimensions of the query tensor
        k = key.permute(0, 2, 1, 3)      # Permute the dimensions of the key tensor
        v = value.permute(0, 2, 1, 3)    # Permute the dimensions of the value tensor
        
        t1 = torch.matmul(q, k.transpose(-2, -1))  # Matrix multiplication
        t2 = t1.div(inv_scale_factor)                # Divide by inverse scale factor
        t3 = t2.softmax(dim=-1)                      # Apply softmax
        t4 = self.dropout(t3)                        # Apply dropout
        output = t4.matmul(v)                        # Matrix multiplication with value tensor
        
        return output

# Initializing the model
dim = 64  # Example dimension
dropout_p = 0.1  # Dropout probability
model = AttentionModel(dim, dropout_p)

# Inputs to the model
batch_size = 2  # Example batch size
query = torch.randn(batch_size, dim, 10, 8)  # (batch_size, dim, seq_length, num_heads)
key = torch.randn(batch_size, dim, 10, 8)    # (batch_size, dim, seq_length, num_heads)
value = torch.randn(batch_size, dim, 10, 8)  # (batch_size, dim, seq_length, num_heads)
inv_scale_factor = torch.tensor(0.1)          # Example inverse scale factor

# Forward pass
output = model(query, key, value, inv_scale_factor)

# Output tensor
print(output.shape)
