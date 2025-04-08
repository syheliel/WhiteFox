import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.inv_scale_factor = embed_size ** 0.5  # Inverse scale factor for scaling

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))  # Compute the dot product of query and key
        scaled_qk = qk / self.inv_scale_factor  # Scale the dot product
        softmax_qk = scaled_qk.softmax(dim=-1)  # Apply softmax
        dropout_qk = self.dropout(softmax_qk)  # Apply dropout
        output = dropout_qk.matmul(value)  # Compute the dot product with value
        return output

# Parameters for the model
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads (not used in this example, but typical in transformers)
dropout_p = 0.1  # Dropout probability

# Initializing the model
model = AttentionModel(embed_size, num_heads, dropout_p)

# Generating input tensors
batch_size = 2  # Number of samples in the batch
seq_length = 10  # Length of the input sequences

query = torch.randn(batch_size, seq_length, embed_size)
key = torch.randn(batch_size, seq_length, embed_size)
value = torch.randn(batch_size, seq_length, embed_size)

# Forward pass through the model
output = model(query, key, value)

# Output tensor shape
print("Output shape:", output.shape)
