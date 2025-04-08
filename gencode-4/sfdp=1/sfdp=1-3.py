import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_size, heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))  # Compute the dot product of query and key tensors
        scaled_qk = qk.div(inv_scale_factor)               # Scale the dot product by an inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1)             # Apply softmax to the scaled dot product
        dropout_qk = self.dropout(softmax_qk)              # Apply dropout to the softmax output
        output = dropout_qk.matmul(value)                   # Compute the dot product of the dropout output and the value tensor
        return output

# Initialize model parameters
embed_size = 64    # Embedding size for query, key, and value
heads = 8         # Number of attention heads
dropout_p = 0.1   # Dropout probability
inv_scale_factor = embed_size ** 0.5  # Inverse scale factor for attention scaling

# Create the model
model = AttentionModel(embed_size, heads, dropout_p)

# Create input tensors
batch_size = 1
seq_length = 10

# Random tensors for query, key, and value
query = torch.randn(batch_size, seq_length, embed_size)
key = torch.randn(batch_size, seq_length, embed_size)
value = torch.randn(batch_size, seq_length, embed_size)

# Pass inputs through the model
output = model(query, key, value, inv_scale_factor)

# Output shape
print("Output shape:", output.shape)
