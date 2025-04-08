import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_dim, dropout_p):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.inv_scale_factor = embed_dim ** 0.5  # Inverse scale factor for scaled dot-product attention

    def forward(self, x):
        # Compute query, key, and value
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Compute dot product of query and key
        qk = torch.matmul(q, k.transpose(-2, -1))

        # Scale the dot product
        scaled_qk = qk / self.inv_scale_factor

        # Apply softmax
        softmax_qk = scaled_qk.softmax(dim=-1)

        # Apply dropout
        dropout_qk = self.dropout(softmax_qk)

        # Compute output
        output = dropout_qk.matmul(v)

        return output

# Initializing the model
embed_dim = 64  # Example embedding dimension
dropout_p = 0.1  # Example dropout probability
model = AttentionModel(embed_dim, dropout_p)

# Sample input tensor (batch_size, sequence_length, embed_dim)
x_input = torch.randn(1, 10, embed_dim)  # Example input with batch size 1 and sequence length 10
output = model(x_input)

print(output.shape)  # Output shape should be (1, 10, embed_dim)
