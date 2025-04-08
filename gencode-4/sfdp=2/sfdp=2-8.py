import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.inv_scale_factor = (embed_size ** 0.5)  # Inverse scale factor for scaling the dot product

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        qk = torch.matmul(query, key.transpose(-2, -1))  # Dot product of query and key
        scaled_qk = qk / self.inv_scale_factor  # Scale the dot product
        softmax_qk = scaled_qk.softmax(dim=-1)  # Apply softmax
        dropout_qk = self.dropout(softmax_qk)  # Apply dropout
        output = dropout_qk.matmul(value)  # Dot product with value tensor
        return output

# Initializing the model
embed_size = 64  # Size of each embedding vector
num_heads = 8    # Number of attention heads (not used in this implementation)
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_size, num_heads, dropout_p)

# Inputs to the model
batch_size = 1
sequence_length = 10  # Length of the input sequence
x = torch.randn(batch_size, sequence_length, embed_size)  # Input tensor

# Forward pass through the model
output = model(x)

# Checking the output shape
print("Output shape:", output.shape)
