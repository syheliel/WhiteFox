import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, dropout_p):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout_p)
    
    def forward(self, x):
        # Assuming x has shape [batch_size, seq_length, embed_dim]
        q = self.query(x)  # Shape: [batch_size, seq_length, embed_dim]
        k = self.key(x)    # Shape: [batch_size, seq_length, embed_dim]
        v = self.value(x)  # Shape: [batch_size, seq_length, embed_dim]
        
        qk = torch.matmul(q, k.transpose(-2, -1))  # Shape: [batch_size, seq_length, seq_length]
        scale_factor = 1.0 / (q.size(-1) ** 0.5)    # Scaling factor
        scaled_qk = qk.mul(scale_factor)             # Scale the dot product
        softmax_qk = scaled_qk.softmax(dim=-1)       # Apply softmax to the scaled dot product
        dropout_qk = self.dropout(softmax_qk)        # Apply dropout to the softmax output
        output = dropout_qk.matmul(v)                # Compute the dot product of the dropout output and the value
        return output

# Initializing the model
embed_dim = 64
dropout_p = 0.1
model = AttentionModel(embed_dim, dropout_p)

# Inputs to the model
batch_size = 2
seq_length = 10
x_input = torch.randn(batch_size, seq_length, embed_dim)

# Forward pass
output = model(x_input)

print("Output shape:", output.shape)  # Should be [batch_size, seq_length, embed_dim]
