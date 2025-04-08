import torch

# Hyperparameters
dropout_p = 0.1
inv_scale_factor = 1.0 / (64 ** 0.5)  # Assuming key dimension is 64

# Model
class AttentionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))  # Compute dot product
        scaled_qk = qk.div(inv_scale_factor)               # Scale the dot product
        softmax_qk = scaled_qk.softmax(dim=-1)             # Apply softmax
        dropout_qk = self.dropout(softmax_qk)              # Apply dropout
        output = dropout_qk.matmul(value)                   # Compute output
        return output

# Initializing the model
model = AttentionModel()

# Inputs to the model
batch_size = 1
num_queries = 10
num_keys_values = 20
embedding_dim = 64

# Random input tensors
query = torch.randn(batch_size, num_queries, embedding_dim)  # Shape: (1, 10, 64)
key = torch.randn(batch_size, num_keys_values, embedding_dim) # Shape: (1, 20, 64)
value = torch.randn(batch_size, num_keys_values, embedding_dim) # Shape: (1, 20, 64)

# Forward pass
output = model(query, key, value)

# Output shape
print("Output shape:", output.shape)
