import torch

# Model Definition
class AttentionModel(torch.nn.Module):
    def __init__(self, query_size, key_size, value_size, dropout_p):
        super().__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.dropout_p = dropout_p
        self.dropout = torch.nn.Dropout(p=self.dropout_p)
        
    def forward(self, query, key, value):
        # Compute attention weights
        attn_weight = torch.bmm(query, key.transpose(1, 2)).softmax(dim=-1)
        
        # Apply dropout to the attention weights
        attn_weight = self.dropout(attn_weight)
        
        # Compute the output
        output = torch.bmm(attn_weight, value)
        return output

# Parameters for the model
query_size = 10  # sequence length of query
key_size = 10    # sequence length of key
value_size = 10  # sequence length of value
dropout_p = 0.1  # dropout probability

# Initialize the model
model = AttentionModel(query_size, key_size, value_size, dropout_p)

# Inputs to the model
# Create random tensors for query, key, and value
batch_size = 2  # number of samples in a batch
query = torch.randn(batch_size, query_size, key_size)  # (batch_size, query_length, key_length)
key = torch.randn(batch_size, key_size, key_size)      # (batch_size, key_length, key_length)
value = torch.randn(batch_size, key_size, value_size)  # (batch_size, key_length, value_length)

# Get the output from the model
output = model(query, key, value)

# Print the output shape for verification
print(output.shape)  # Should be (batch_size, query_length, value_size)
