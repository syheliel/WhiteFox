import torch

# Description of the model
class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super().__init__()
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.inv_scale_factor = embed_dim ** 0.5  # Inverse scale factor for attention

    def forward(self, query, key, value):
        qk = torch.matmul(self.query(query), self.key(key).transpose(-2, -1))  # Compute dot product
        scaled_qk = qk / self.inv_scale_factor  # Scale the dot product
        softmax_qk = scaled_qk.softmax(dim=-1)  # Apply softmax
        dropout_qk = self.dropout(softmax_qk)  # Apply dropout
        output = torch.matmul(dropout_qk, self.value(value))  # Compute output
        return output

# Initializing the model
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_dim, num_heads, dropout_p)

# Generating input tensors for the model
batch_size = 2  # Number of samples in a batch
seq_length = 10  # Sequence length
query_input = torch.randn(batch_size, seq_length, embed_dim)  # Query tensor
key_input = torch.randn(batch_size, seq_length, embed_dim)    # Key tensor
value_input = torch.randn(batch_size, seq_length, embed_dim)  # Value tensor

# Forward pass through the model
output = model(query_input, key_input, value_input)
print(output.shape)  # Output tensor shape
