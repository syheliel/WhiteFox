import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, dropout_p):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dropout_p = dropout_p

    def forward(self, query, key, value):
        # Compute attention weights
        attn_weight = torch.bmm(query, key.transpose(1, 2)).softmax(dim=-1)
        # Apply dropout to the attention weights
        attn_weight = torch.nn.functional.dropout(attn_weight, p=self.dropout_p)
        # Compute the output
        output = torch.bmm(attn_weight, value)
        return output

# Initialize the model
query_dim = 64  # Dimension of the query
key_dim = 64    # Dimension of the key
value_dim = 64  # Dimension of the value
dropout_p = 0.1 # Dropout probability

model = AttentionModel(query_dim, key_dim, value_dim, dropout_p)

# Inputs to the model
batch_size = 2  # Number of samples in a batch
seq_length = 10 # Length of the sequence

# Create random input tensors for query, key, and value
query = torch.randn(batch_size, seq_length, query_dim)
key = torch.randn(batch_size, seq_length, key_dim)
value = torch.randn(batch_size, seq_length, value_dim)

# Get the output from the model
output = model(query, key, value)

# Print the output shape
print(output.shape)  # Expected shape: (batch_size, seq_length, value_dim)
