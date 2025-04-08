import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super().__init__()
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.inv_scale_factor = embed_dim ** 0.5  # Inverse scale factor for scaling the dot product

    def forward(self, x):
        q = self.query(x)  # Query
        k = self.key(x)    # Key
        v = self.value(x)  # Value

        qk = torch.matmul(q, k.transpose(-2, -1))  # Compute dot product of query and key
        scaled_qk = qk.div(self.inv_scale_factor)    # Scale by inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1)       # Apply softmax to scaled dot product
        dropout_qk = self.dropout(softmax_qk)        # Apply dropout
        output = dropout_qk.matmul(v)                # Compute dot product with value tensor
        return output

# Model initialization
embed_dim = 64   # Example embedding dimension
num_heads = 8    # Example number of heads (not used directly here)
dropout_p = 0.1  # Example dropout probability
model = AttentionModel(embed_dim, num_heads, dropout_p)

# Generating the input tensor (batch_size, sequence_length, embed_dim)
batch_size = 1
sequence_length = 10  # Example sequence length
x_input = torch.randn(batch_size, sequence_length, embed_dim)

# Forward pass through the model
output = model(x_input)
print(output.shape)  # Output shape
