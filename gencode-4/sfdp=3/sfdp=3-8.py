import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, dropout_p):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        q = self.query(x)  # Compute query
        k = self.key(x)    # Compute key
        v = self.value(x)  # Compute value
        
        # Compute the dot product of the query and the transposed key
        qk = torch.matmul(q, k.transpose(-2, -1))
        
        # Scale the dot product
        scale_factor = 1 / (k.size(-1) ** 0.5)  # Scaling by the square root of key dimension
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
# Assume a batch size of 1 and sequence length of 10 for demonstration
x_input = torch.randn(1, 10, embed_dim)  # Input tensor with shape (batch_size, seq_length, embed_dim)
output = model(x_input)

print("Output shape:", output.shape)  # Should be (1, 10, embed_dim)
