import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.inv_scale_factor = embed_size ** 0.5  # Typically, this is sqrt(d_k) where d_k is the dimension of keys

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        qk = torch.matmul(q, k.transpose(-2, -1))  # Compute dot product
        scaled_qk = qk / self.inv_scale_factor  # Scale the dot product
        softmax_qk = scaled_qk.softmax(dim=-1)  # Apply softmax
        dropout_qk = self.dropout(softmax_qk)  # Apply dropout
        output = dropout_qk.matmul(v)  # Compute the final output
        return output

# Model parameters
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads (not used here but often relevant in multi-head attention)
dropout_p = 0.1  # Dropout probability

# Initializing the model
model = AttentionModel(embed_size, num_heads, dropout_p)

# Input tensor
batch_size = 1
sequence_length = 10  # Length of the input sequence
x = torch.randn(batch_size, sequence_length, embed_size)  # Random input tensor

# Forward pass
output = model(x)

# Output shape
print("Output shape:", output.shape)
