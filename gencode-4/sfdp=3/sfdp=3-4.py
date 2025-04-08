import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, dropout_p):
        super().__init__()
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)
        self.scale_factor = 1.0 / (embed_dim ** 0.5)  # Scaling factor for dot product
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, x):
        # Assume x is of shape (batch_size, seq_len, embed_dim)
        q = self.query(x)  # Query
        k = self.key(x)    # Key
        v = self.value(x)  # Value
        
        # Compute attention
        qk = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, seq_len, seq_len)
        scaled_qk = qk.mul(self.scale_factor)        # Scale the dot product
        softmax_qk = scaled_qk.softmax(dim=-1)      # Apply softmax to the scaled dot product
        dropout_qk = self.dropout(softmax_qk)       # Apply dropout to the softmax output
        
        output = dropout_qk.matmul(v)               # Compute the dot product of the dropout output and the value
        return output

# Initializing the model
embed_dim = 64
dropout_p = 0.1
model = AttentionModel(embed_dim, dropout_p)

# Inputs to the model
batch_size = 1
seq_len = 10
x_input = torch.randn(batch_size, seq_len, embed_dim)

# Get the model's output
output = model(x_input)

print(output.shape)  # Output shape should be (batch_size, seq_len, embed_dim)
