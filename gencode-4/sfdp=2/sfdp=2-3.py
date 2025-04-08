import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, dropout_p):
        super(AttentionModel, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.inv_scale_factor = embed_dim ** 0.5  # Typically the square root of the dimension

    def forward(self, x):
        q = self.query(x)  # (batch_size, seq_length, embed_dim)
        k = self.key(x)    # (batch_size, seq_length, embed_dim)
        v = self.value(x)  # (batch_size, seq_length, embed_dim)

        qk = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, seq_length, seq_length)
        scaled_qk = qk / self.inv_scale_factor      # Scale the dot product
        softmax_qk = scaled_qk.softmax(dim=-1)      # Apply softmax
        dropout_qk = self.dropout(softmax_qk)        # Apply dropout
        output = torch.matmul(dropout_qk, v)        # Compute the dot product with value tensor

        return output

# Initializing the model
embed_dim = 64  # Embedding dimension
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_dim, dropout_p)

# Inputs to the model
batch_size = 8
seq_length = 10
input_tensor = torch.randn(batch_size, seq_length, embed_dim)  # Random input tensor

# Forward pass through the model
output = model(input_tensor)

print(output.shape)  # Output shape should be (batch_size, seq_length, embed_dim)
